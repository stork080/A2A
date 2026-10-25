"""
Invoice action agent - A2A 1.0 HTTP+JSON.

Layers (kept separate on purpose):
  1. protocol  : routes, auth, headers, envelopes
  2. storage   : SQLite (tasks, message idempotency, decision cache)
  3. reasoning : one batched LLM call, code-validated, cached by package content

Run:  uvicorn main:app --host 0.0.0.0 --port $PORT      (ONE worker only)
"""
import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("agent")

# --------------------------------------------------------------------------
# Config (all via environment variables)
# --------------------------------------------------------------------------
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").strip()  # EXACT url you submit
BASE_PATH = urlparse(PUBLIC_BASE_URL).path.rstrip("/") if PUBLIC_BASE_URL else ""
LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip().strip("\"'")
LLM_BASE_URL = (
    os.environ.get("LLM_BASE_URL", "").strip().strip("\"'")
    or "https://generativelanguage.googleapis.com/v1beta/openai"
)
if not LLM_BASE_URL.lower().startswith(("http://", "https://")):
    LLM_BASE_URL = "https://" + LLM_BASE_URL
LLM_BASE_URL = LLM_BASE_URL.rstrip("/")
if LLM_BASE_URL.endswith("/chat/completions"):
    LLM_BASE_URL = LLM_BASE_URL[: -len("/chat/completions")]
LLM_MODEL = os.environ.get("LLM_MODEL", "").strip().strip("\"'") or "gemini-2.5-flash"
# 0 = return evidence refs exactly as written, brackets included, e.g. "[PO-123]"
# 1 = strip the brackets, e.g. "PO-123"   (flip this if evidence checks fail)
STRIP_BRACKETS = os.environ.get("STRIP_BRACKETS", "0") == "1"
# Optional: restrict to specific tokens. Empty = every non-empty Bearer token is a separate user.
ALLOWED_TOKENS = {t.strip() for t in os.environ.get("ALLOWED_TOKENS", "").split(",") if t.strip()}
DATA_DIR = os.environ.get("DATA_DIR", "./data")
DECISION_BUDGET_SECONDS = 40
LLM_CHUNK = max(1, int(os.environ.get("LLM_CHUNK", "6") or 6))  # packages per model call  # each request has 45s

A2A_JSON = "application/a2a+json"
MT_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
MT_PROPOSALS = "application/vnd.ga5.invoice-action-proposals+json"
MT_RESULTS = "application/vnd.ga5.invoice-action-results+json"
MT_RECEIPTS = "application/vnd.ga5.invoice-action-receipts+json"
ACTIONS = ["settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"]
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}
REF_RE = re.compile(r"\[[^\[\]\n]{1,120}\]")
MAX_RESPONSE_BYTES = 480 * 1024


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def canon(obj):
    """Recursively key-sorted compact JSON."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def respond(obj, status=200):
    return Response(
        content=json.dumps(obj, separators=(",", ":"), ensure_ascii=False),
        status_code=status,
        media_type=A2A_JSON,
        headers={"A2A-Version": "1.0", "Cache-Control": "no-store"},
    )


def err(status, code, message="Request could not be processed."):
    # Generic on purpose: never echo ids or reveal whether another user's task exists.
    return respond({"error": {"code": code, "message": message}}, status)


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------
os.makedirs(DATA_DIR, exist_ok=True)
_db = sqlite3.connect(os.path.join(DATA_DIR, "agent.db"), check_same_thread=False, isolation_level=None)
_db_lock = threading.Lock()
with _db_lock:
    _db.execute("PRAGMA journal_mode=WAL")
    _db.execute("PRAGMA synchronous=FULL")
    _db.execute(
        "CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, principal TEXT NOT NULL,"
        " created REAL NOT NULL, body TEXT NOT NULL)"
    )
    _db.execute("CREATE INDEX IF NOT EXISTS tasks_principal ON tasks(principal, created)")
    _db.execute(
        "CREATE TABLE IF NOT EXISTS messages (principal TEXT NOT NULL, message_id TEXT NOT NULL,"
        " hash TEXT NOT NULL, task_id TEXT NOT NULL, PRIMARY KEY (principal, message_id))"
    )
    _db.execute("CREATE TABLE IF NOT EXISTS decisions (key TEXT PRIMARY KEY, body TEXT NOT NULL)")


def db_all(sql, args=()):
    with _db_lock:
        return _db.execute(sql, args).fetchall()


def load_task(principal, task_id):
    rows = db_all("SELECT body FROM tasks WHERE id=? AND principal=?", (task_id, principal))
    return json.loads(rows[0][0]) if rows else None


def save_task(principal, task, message_id=None, message_hash=None):
    """Task row and idempotency record are written in ONE transaction."""
    with _db_lock:
        _db.execute("BEGIN IMMEDIATE")
        try:
            _db.execute(
                "INSERT INTO tasks (id, principal, created, body) VALUES (?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET body=excluded.body",
                (task["id"], principal, datetime.now().timestamp(), json.dumps(task)),
            )
            if message_id is not None:
                _db.execute(
                    "INSERT OR IGNORE INTO messages (principal, message_id, hash, task_id) VALUES (?,?,?,?)",
                    (principal, message_id, message_hash, task["id"]),
                )
            _db.execute("COMMIT")
        except Exception:
            _db.execute("ROLLBACK")
            raise


_locks = {}


def lock_for(key):
    if key not in _locks:  # single event loop => no race creating the lock
        _locks[key] = asyncio.Lock()
    return _locks[key]


# --------------------------------------------------------------------------
# Reasoning layer
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a meticulous accounts-payable auditor. For EACH invoice package choose exactly ONE action:
- settle_invoice: invoice is valid, reconciled, and within the autonomous (delegated) payment authority.
- request_approval: commercially valid and reconciled, but the amount is OUTSIDE the delegated authority/limit.
- hold_invoice: payment must pause until a stated verification completes (e.g. bank-detail callback, missing receipt/confirmation, compliance check).
- reject_duplicate: the SAME commercial invoice (same vendor, same invoice number, same amount) was already PAID.
- open_exception: material records conflict (PO vs invoice vs receipt: amounts, quantities, vendor identity, bank details) and need an exception workflow.

How to read a package:
1. Each package is split into numbered paragraphs P1, P2, ... Exactly ONE paragraph is the controlling paragraph: the one that states the current facts that determine the action for the invoice being processed.
2. IGNORE the cover-sheet summary, archive/historical/old examples, training or sample material, template wording, and quoted past cases. Action words (hold, reject, approve, settle, duplicate, exception) inside those places, or in negated form ("not a duplicate", "hold released", "no conflict"), decide nothing.
3. A payment for a DIFFERENT invoice number is not a duplicate, even for the same vendor, PO or amount.
4. Compare the invoice amount with the delegated/autonomous limit stated in the documents or policy. Over the limit => request_approval, even when everything else is valid.
5. amountMinor is an INTEGER in minor units of the invoice currency (1234.50 INR -> 123450; use 1 per unit for currencies without minor units such as JPY). Use the amount of the invoice being decided.
6. vendorName and invoiceNumber must be copied exactly as written in the documents.

Reply with JSON ONLY, no markdown:
{"decisions":[{"packageId":"...","reasoning":"2-3 sentences: the controlling facts, what you ignored, why this action","controllingParagraph":"P3","action":"<one of the five>","vendorName":"...","invoiceNumber":"...","amountMinor":12345,"currency":"INR","evidenceRefs":["[REF-1]","[REF-2]","[REF-3]"]}]}
Write "reasoning" BEFORE choosing the action. evidenceRefs are the bracketed references copied verbatim from the controlling paragraph."""


def split_paragraphs(pkg):
    """Flatten a package into [(path, text)] paragraphs."""
    out = []

    def walk(x, path):
        if isinstance(x, str):
            chunks = re.split(r"\n\s*\n", x.strip())
            if len(chunks) == 1 and "\n" in chunks[0]:
                chunks = chunks[0].split("\n")
            for c in chunks:
                if c.strip():
                    out.append((path, c.strip()))
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            out.append((path, str(x)))

    walk({k: v for k, v in pkg.items() if k != "packageId"}, "")
    return out


def pkg_content_key(pkg, policy_rev):
    body = {k: v for k, v in pkg.items() if k not in ("packageId", "id")}
    return sha(canon({"p": body, "rev": policy_rev}))


def render_package(pid, paras):
    lines = [f"=== PACKAGE {pid} ==="]
    for i, (path, text) in enumerate(paras, 1):
        lines.append(f"P{i} ({path}): {text}")
    return "\n".join(lines)


def uniq(seq):
    seen, out = set(), []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def to_int(v):
    try:
        if isinstance(v, bool):
            return None
        return int(round(float(str(v).replace(",", "").strip())))
    except Exception:
        return None


def build_decision(d, paras, policy_rev, strict=True, pid=""):
    """Validate one LLM decision. strict=False keeps the model's action even if details are imperfect."""
    if not isinstance(d, dict):
        log.warning("pkg %s: no decision returned by model", pid)
        return None
    if d.get("action") not in ACTIONS:
        log.warning("pkg %s: invalid action %r", pid, d.get("action"))
        return None
    all_text = "\n".join(t for _, t in paras)
    m = re.search(r"\d+", str(d.get("controllingParagraph", "")))
    pnum = int(m.group()) if m and 1 <= int(m.group()) <= len(paras) else None
    cand = uniq(REF_RE.findall(paras[pnum - 1][1])) if pnum else []
    if len(cand) != 3:
        alt = [r if r.startswith("[") else f"[{r}]" for r in d.get("evidenceRefs", []) if isinstance(r, str)]
        alt = [r for r in uniq(alt) if r in all_text]
        if len(alt) >= 2:
            cand = alt[:3]
        elif strict and len(cand) < 2:
            log.warning("pkg %s: could not determine 3 evidence refs (para=%s)", pid, pnum)
            return None
        else:
            cand = (cand or alt)[:3]
    amount = to_int(d.get("amountMinor"))
    vendor = str(d.get("vendorName", "")).strip()
    inv = str(d.get("invoiceNumber", "")).strip()
    cur = str(d.get("currency", "")).strip().upper()
    if strict and (amount is None or not vendor or not inv or not cur):
        log.warning("pkg %s: incomplete facts %s", pid, {k: d.get(k) for k in ("vendorName", "invoiceNumber", "amountMinor", "currency")})
        return None
    action = d["action"]
    reasoning = re.sub(r"\s+", " ", str(d.get("reasoning", "")).strip())
    ev = f" supported by evidence {', '.join(cand)}" if cand else " based on the controlling paragraph"
    rationale = (
        f"Chosen action: {action},{ev}. {reasoning} "
        f"The controlling paragraph (P{pnum}) was used under policy revision {policy_rev}; "
        "archive examples, cover-sheet text and negated action words were disregarded."
    )[:1500]
    refs = [r[1:-1].strip() for r in cand] if STRIP_BRACKETS else cand
    return {
        "action": action,
        "facts": {
            "vendorName": vendor or "unknown",
            "invoiceNumber": inv or "unknown",
            "amountMinor": amount if amount is not None else 0,
            "currency": cur or "INR",
        },
        "evidenceRefs": refs,
        "rationale": rationale,
    }


def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        a, b = text.find("{"), text.rfind("}")
        return json.loads(text[a : b + 1])


async def call_llm(user_prompt, timeout=38.0):
    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY is not set (check Render > Environment)")
    body = {
        "model": LLM_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
    url = LLM_BASE_URL.rstrip("/") + "/chat/completions"
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(4):
            r = await client.post(url, json=body, headers=headers)
            if r.status_code == 400 and "response_format" in body:
                body.pop("response_format")
                continue
            if r.status_code in (429, 500, 502, 503) and attempt < 3:
                await asyncio.sleep(3 * (attempt + 1))  # 3s, 6s, 9s backoff
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"LLM HTTP {r.status_code} from {url} model={LLM_MODEL}: {r.text[:300]}")
            return r.json()["choices"][0]["message"]["content"]
        raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:300]}")


async def llm_decide(items, policy_ctx, policy_rev, results):
    """items: [(pid, paras)] -> fills results[pid]. Parallel chunks of 4, then lenient single retries."""
    header = f"Policy revision: {policy_rev}\nOther batch fields: {canon(policy_ctx)}\n\n"

    async def run(batch_items, strict):
        prompt = header + "\n\n".join(render_package(pid, paras) for pid, paras in batch_items)
        text = await call_llm(prompt)
        log.info("LLM raw output (first 500 chars): %s", text[:500].replace("\n", " "))
        parsed = parse_json(text).get("decisions", [])
        by_id = {str(d.get("packageId")): d for d in parsed if isinstance(d, dict)}
        for pid, paras in batch_items:
            dec = build_decision(by_id.get(pid), paras, policy_rev, strict, pid)
            if dec:
                results[pid] = dec

    chunks = [items[i : i + LLM_CHUNK] for i in range(0, len(items), LLM_CHUNK)]
    outs = await asyncio.gather(*(run(c, True) for c in chunks), return_exceptions=True)
    for o in outs:
        if isinstance(o, Exception):
            log.warning("LLM chunk failed: %r", o)
    missing = [(pid, paras) for pid, paras in items if pid not in results]
    if missing:
        log.info("retrying %d package(s) individually (lenient)", len(missing))
        outs = await asyncio.gather(*(run([it], False) for it in missing), return_exceptions=True)
        for o in outs:
            if isinstance(o, Exception):
                log.warning("LLM single retry failed: %r", o)


async def decide(batch):
    packages = batch["packages"]
    policy_rev = batch.get("policyRevision", "")
    policy_ctx = {k: v for k, v in batch.items() if k not in ("packages", "batchId")}
    decisions, todo = {}, []
    for pkg in packages:
        pid = str(pkg["packageId"])
        key = pkg_content_key(pkg, policy_rev)
        row = db_all("SELECT body FROM decisions WHERE key=?", (key,))
        if row:
            decisions[pid] = json.loads(row[0][0])
        else:
            todo.append((pid, split_paragraphs(pkg), key))
    fresh = {}
    if todo:
        try:
            await asyncio.wait_for(
                llm_decide([(p, pa) for p, pa, _ in todo], policy_ctx, policy_rev, fresh),
                timeout=DECISION_BUDGET_SECONDS,
            )
        except Exception as e:
            log.warning("decision step incomplete: %s", e)
    for pid, paras, key in todo:
        if pid in fresh:
            decisions[pid] = fresh[pid]
            with _db_lock:
                _db.execute("INSERT OR REPLACE INTO decisions (key, body) VALUES (?,?)", (key, json.dumps(fresh[pid])))
        else:
            # Never pay when the model failed; not cached so a later run retries properly.
            refs = uniq(REF_RE.findall("\n".join(t for _, t in paras)))[:3]
            decisions[pid] = {
                "action": "hold_invoice",
                "facts": {"vendorName": "unknown", "invoiceNumber": "unknown", "amountMinor": 0, "currency": "INR"},
                "evidenceRefs": [r[1:-1] for r in refs] if STRIP_BRACKETS else refs,
                "rationale": "Chosen action: hold_invoice because automated analysis could not be completed; "
                "payment is paused until the invoice is reviewed manually with the cited evidence.",
            }
    proposals = []
    for pkg in packages:
        pid = str(pkg["packageId"])
        d = decisions[pid]
        proposals.append(
            {
                "packageId": pkg["packageId"],
                "actionId": "act-" + uuid.uuid4().hex[:20],
                "action": d["action"],
                "facts": d["facts"],
                "evidenceRefs": d["evidenceRefs"],
                "rationale": d["rationale"],
            }
        )
    return {"batchId": batch["batchId"], "proposals": proposals}


# --------------------------------------------------------------------------
# Protocol layer
# --------------------------------------------------------------------------
app = FastAPI(redirect_slashes=False, docs_url=None, redoc_url=None, openapi_url=None)
router = APIRouter()


def base_url(request):
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "localhost")
    return f"https://{host}/"


@app.get("/.well-known/agent-card.json")
async def agent_card(request: Request):
    return respond(
        {
            "name": "Invoice Action Agent",
            "description": "Reads invoice packages, proposes one evidence-backed business action per "
            "invoice, and executes only the proposals the grader accepts.",
            "version": "1.0.0",
            "supportedInterfaces": [
                {"url": base_url(request), "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}
            ],
            "capabilities": {"streaming": False, "pushNotifications": False},
            "defaultInputModes": [MT_BATCH, MT_RESULTS],
            "defaultOutputModes": [MT_PROPOSALS, MT_RECEIPTS],
            "skills": [
                {
                    "id": "invoice_action_agent",
                    "name": "Invoice action agent",
                    "description": "Chooses settle, approval, hold, duplicate-rejection or exception "
                    "actions for invoice packages and cites the decisive evidence.",
                    "tags": ["invoice", "accounts-payable", "approval", "duplicate-detection"],
                }
            ],
        }
    )


@app.get("/healthz")
async def healthz():
    return Response("ok", media_type="text/plain")


@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return Response('{"status":"ok","agent":"Invoice Action Agent"}', media_type="application/json")


def check_request(request, is_post, require_ct=False):
    m = re.match(r"(?i)^bearer\s+(\S+)$", request.headers.get("authorization", "").strip())
    if not m or (ALLOWED_TOKENS and m.group(1) not in ALLOWED_TOKENS):
        return None, err(401, "UNAUTHENTICATED", "Authentication required.")
    if request.headers.get("a2a-version", "").strip() != "1.0":
        return None, err(400, "VERSION_NOT_SUPPORTED", "Only A2A-Version 1.0 is supported.")
    accept = request.headers.get("accept", "")
    if accept and not any(t in accept.lower() for t in ("a2a+json", "application/json", "*/*", "application/*")):
        return None, err(406, "NOT_ACCEPTABLE", "Use application/a2a+json.")
    if is_post:
        ct = request.headers.get("content-type")
        if ct is None:
            if require_ct:
                return None, err(415, "UNSUPPORTED_MEDIA_TYPE", "Content-Type must be application/a2a+json.")
        elif ct.split(";")[0].strip().lower() != A2A_JSON:
            return None, err(415, "UNSUPPORTED_MEDIA_TYPE", "Content-Type must be application/a2a+json.")
    return sha(m.group(1))[:32], None


def find_part(message, media_type):
    parts = [p for p in message.get("parts", []) if isinstance(p, dict) and p.get("mediaType") == media_type]
    return parts


def apply_results(task, message):
    """Return (executions, None) or (None, reason). Pure validation, no mutation."""
    parts = find_part(message, MT_RESULTS)
    if len(parts) != 1 or not isinstance(parts[0].get("data"), dict):
        return None, "bad results part"
    data = parts[0]["data"]
    stored = task["artifacts"][0]["parts"][0]["data"]
    if data.get("batchId") != stored["batchId"]:
        return None, "batch mismatch"
    results = data.get("results")
    if not isinstance(results, list) or not results:
        return None, "no results"
    by_pkg = {p["packageId"]: p for p in stored["proposals"]}
    seen, executions = set(), []
    for r in results:
        if not isinstance(r, dict):
            return None, "bad result"
        prop = by_pkg.get(r.get("packageId"))
        if prop is None or r["packageId"] in seen:
            return None, "unknown or repeated package"
        seen.add(r["packageId"])
        if r.get("actionId") != prop["actionId"] or r.get("action") != prop["action"]:
            return None, "action mismatch"
        outcome = r.get("outcome")
        if outcome not in ("ACCEPTED", "REJECTED"):
            return None, "bad outcome"
        if outcome == "ACCEPTED":
            nonce = r.get("receiptNonce")
            if not isinstance(nonce, str) or not nonce:
                return None, "missing receipt nonce"
            executions.append(
                {
                    "packageId": prop["packageId"],
                    "actionId": prop["actionId"],
                    "action": prop["action"],
                    "receiptNonce": nonce,
                    "facts": prop["facts"],
                    "evidenceRefs": prop["evidenceRefs"],
                }
            )
    return executions, None


@router.post("/message:send")
async def message_send(request: Request):
    principal, bad = check_request(request, True, True)
    if bad:
        return bad
    try:
        body = json.loads((await request.body()) or b"{}")
        message = body["message"]
        message_id = message["messageId"]
        assert isinstance(message, dict) and isinstance(message_id, str) and message_id
        assert message.get("role") == "ROLE_USER" and isinstance(message.get("parts"), list)
    except Exception:
        return err(400, "INVALID_REQUEST", "Malformed message.")
    mhash = sha(canon(message))

    async with lock_for(("msg", principal, message_id)):
        rec = db_all("SELECT hash, task_id FROM messages WHERE principal=? AND message_id=?", (principal, message_id))
        if rec:
            if rec[0][0] != mhash:
                return err(409, "IDEMPOTENCY_CONFLICT", "messageId was already used with different content.")
            task = load_task(principal, rec[0][1])
            return respond({"task": task}) if task else err(404, "NOT_FOUND")

        # ---- continuation: results for an existing task -------------------
        if message.get("taskId"):
            if not isinstance(message["taskId"], str):
                return err(400, "INVALID_REQUEST")
            async with lock_for(("task", message["taskId"])):
                task = load_task(principal, message["taskId"])  # other users' tasks look nonexistent
                if task is None:
                    return err(404, "NOT_FOUND")
                if task["status"]["state"] in TERMINAL:
                    return err(409, "TASK_TERMINAL", "Task can no longer change.")
                if message.get("contextId") != task["contextId"]:
                    return err(400, "INVALID_CONTINUATION")
                executions, reason = apply_results(task, message)
                if reason:
                    log.info("continuation rejected: %s", reason)
                    return err(400, "INVALID_CONTINUATION")
                stored = task["artifacts"][0]["parts"][0]["data"]
                task["artifacts"].append(
                    {
                        "artifactId": "receipts-" + task["id"][:8],
                        "name": "invoice-action-receipts",
                        "parts": [
                            {"mediaType": MT_RECEIPTS, "data": {"batchId": stored["batchId"], "executions": executions}}
                        ],
                    }
                )
                task["history"].append(message)
                task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now_iso()}
                save_task(principal, task, message_id, mhash)
                return respond({"task": task})

        # ---- new batch -----------------------------------------------------
        parts = find_part(message, MT_BATCH)
        try:
            batch = parts[0]["data"]
            assert isinstance(batch["batchId"], str) and isinstance(batch["packages"], list) and batch["packages"]
            ids = [str(p["packageId"]) for p in batch["packages"]]
            assert len(set(ids)) == len(ids)
        except Exception:
            return err(400, "INVALID_REQUEST", "Expected one invoice claim batch part.")
        log.info("new batch %s with %d packages; first package: %s", batch["batchId"], len(ids),
                 canon(batch["packages"][0])[:6000])
        payload = await decide(batch)
        task_id = str(uuid.uuid4())
        task = {
            "id": task_id,
            "contextId": message.get("contextId") or str(uuid.uuid4()),
            "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now_iso()},
            "artifacts": [
                {
                    "artifactId": "proposals-" + task_id[:8],
                    "name": "invoice-action-proposals",
                    "parts": [{"mediaType": MT_PROPOSALS, "data": payload}],
                }
            ],
            "history": [message],
        }
        save_task(principal, task, message_id, mhash)
        return respond({"task": task})


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    principal, bad = check_request(request, False)
    if bad:
        return bad
    task = load_task(principal, task_id)
    return respond(task) if task else err(404, "NOT_FOUND")


@router.get("/tasks")
async def list_tasks(request: Request):
    principal, bad = check_request(request, False)
    if bad:
        return bad
    rows = db_all("SELECT body FROM tasks WHERE principal=? ORDER BY created DESC", (principal,))
    tasks, size = [], 0
    for (b,) in rows:
        t = json.loads(b)
        t.pop("history", None)  # keep the list small (512 KiB response cap)
        size += len(json.dumps(t))
        if size > MAX_RESPONSE_BYTES:
            break
        tasks.append(t)
    return respond({"tasks": tasks})


@router.post("/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    principal, bad = check_request(request, True)
    if bad:
        return bad
    async with lock_for(("task", task_id)):
        task = load_task(principal, task_id)
        if task is None:
            return err(404, "NOT_FOUND")
        if task["status"]["state"] in TERMINAL:
            return err(409, "TASK_NOT_CANCELABLE", "Task can no longer change.")
        task["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now_iso()}
        save_task(principal, task)
        return respond(task)


app.include_router(router, prefix=BASE_PATH)
