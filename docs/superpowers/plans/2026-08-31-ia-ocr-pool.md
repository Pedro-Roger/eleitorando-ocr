# IA OCR Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the EasyOCR-first worker with an AI-only asynchronous model pool that stores editable review drafts and requires manual confirmation before final persistence.

**Architecture:** Keep the existing FastAPI service and `ocr_jobs` queue table, but move processing to an in-memory OpenRouter model pool with slot cooldowns and retries. Persist extraction artifacts inside the existing `result` JSON payload so the review and confirmation flow can ship without a database migration.

**Tech Stack:** Python 3.12, FastAPI, unittest, unittest.mock, psycopg2, urllib.request, OpenRouter chat completions API

---

### Task 1: Add tests for model pool selection and AI result handling

**Files:**
- Create: `tests/test_model_pool.py`
- Test: `python3 -m unittest tests/test_model_pool.py -v`

- [ ] **Step 1: Write the failing test**

```python
import unittest
import main


class ModelPoolTests(unittest.TestCase):
    def test_selects_first_available_slot(self):
        slots = [
            main.ModelSlot(model="a", cooldown_until=0.0, consecutive_failures=0, last_error=""),
            main.ModelSlot(model="b", cooldown_until=9999999999.0, consecutive_failures=0, last_error=""),
        ]

        picked = main.pick_available_slot(slots, now=100.0)

        self.assertEqual(picked.model, "a")

    def test_cooldown_moves_slot_out_of_rotation(self):
        slot = main.ModelSlot(model="a", cooldown_until=0.0, consecutive_failures=0, last_error="")

        main.mark_slot_failure(slot, "rate_limit", now=100.0, cooldown_seconds=30.0)

        self.assertGreater(slot.cooldown_until, 100.0)
        self.assertEqual(slot.consecutive_failures, 1)
        self.assertEqual(slot.last_error, "rate_limit")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_model_pool.py -v`
Expected: `AttributeError` or import failure because `ModelSlot`, `pick_available_slot`, and `mark_slot_failure` do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass


@dataclass
class ModelSlot:
    model: str
    cooldown_until: float = 0.0
    consecutive_failures: int = 0
    last_error: str = ""


def pick_available_slot(slots, now=None):
    now = time.time() if now is None else now
    for slot in slots:
        if slot.cooldown_until <= now:
            return slot
    return None


def mark_slot_failure(slot, reason, now=None, cooldown_seconds=30.0):
    now = time.time() if now is None else now
    slot.cooldown_until = now + cooldown_seconds
    slot.consecutive_failures += 1
    slot.last_error = reason
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_model_pool.py -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add tests/test_model_pool.py main.py
git commit -m "test: cover model pool slot state"
```

### Task 2: Add tests for review and confirmation API flow

**Files:**
- Create: `tests/test_review_api.py`
- Test: `python3 -m unittest tests/test_review_api.py -v`

- [ ] **Step 1: Write the failing test**

```python
import json
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
import main


class ReviewApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    @patch("main.fetch_job_row")
    @patch("main.update_job_result_payload")
    def test_patch_review_saves_review_draft(self, update_job_result_payload, fetch_job_row):
        fetch_job_row.return_value = {
            "id": 1,
            "status": "review_required",
            "result": json.dumps({"reviewDraft": {"nome": "A"}}),
        }

        response = self.client.patch("/jobs/1/review", json={"reviewDraft": {"nome": "B"}})

        self.assertEqual(response.status_code, 200)
        payload = update_job_result_payload.call_args.args[1]
        self.assertEqual(payload["reviewDraft"]["nome"], "B")

    @patch("main.fetch_job_row")
    @patch("main.mark_job_confirmed")
    def test_confirm_requires_review_draft(self, mark_job_confirmed, fetch_job_row):
        fetch_job_row.return_value = {
            "id": 1,
            "status": "review_required",
            "result": json.dumps({}),
        }

        response = self.client.post("/jobs/1/confirm")

        self.assertEqual(response.status_code, 400)
        mark_job_confirmed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_review_api.py -v`
Expected: route or helper lookup failure because the review endpoints and helper functions do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def fetch_job_row(job_id):
    ...


def update_job_result_payload(job_id, payload, status=None):
    ...


def mark_job_confirmed(job_id, payload):
    update_job_result_payload(job_id, payload, status="confirmed")


@app.patch("/jobs/{job_id}/review")
async def save_review(job_id: int, body: dict):
    ...


@app.post("/jobs/{job_id}/confirm")
async def confirm_job(job_id: int):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_review_api.py -v`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add tests/test_review_api.py main.py
git commit -m "feat: add review and confirm job endpoints"
```

### Task 3: Implement AI-only model pool worker and Discord alerts

**Files:**
- Modify: `main.py`
- Test: `python3 -m unittest tests/test_model_pool.py -v`

- [ ] **Step 1: Extend the failing tests for request execution**

```python
from urllib.error import HTTPError
from unittest.mock import Mock, patch

    @patch("main.send_discord_alert")
    def test_http_429_puts_slot_in_cooldown_and_alerts(self, send_discord_alert):
        slot = main.ModelSlot(model="a", cooldown_until=0.0, consecutive_failures=0, last_error="")
        err = HTTPError("http://x", 429, "rate limit", hdrs=None, fp=None)

        with self.assertRaises(HTTPError):
            main.handle_model_exception(slot, err, now=100.0)

        self.assertEqual(slot.last_error, "http_429")
        send_discord_alert.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_model_pool.py -v`
Expected: missing `handle_model_exception` or missing alert hook.

- [ ] **Step 3: Write minimal implementation**

```python
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")


def send_discord_alert(event, details):
    if not DISCORD_WEBHOOK_URL:
        return
    ...


def handle_model_exception(slot, exc, now=None):
    code = getattr(exc, "code", None)
    if code == 429:
        mark_slot_failure(slot, "http_429", now=now, cooldown_seconds=60.0)
        send_discord_alert("provider_rate_limit", {"model": slot.model, "code": code})
    else:
        mark_slot_failure(slot, "provider_error", now=now, cooldown_seconds=20.0)
        send_discord_alert("provider_error", {"model": slot.model, "code": code})
    raise exc
```

- [ ] **Step 4: Replace the worker’s processing path**

```python
def process_image_ai_only(img):
    payload = build_review_payload(img)
    return payload


async def ocr_worker():
    ...
    result = await asyncio.to_thread(process_image_ai_only, img)
    update_job_result_payload(job_id, result, status="review_required")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m unittest tests/test_model_pool.py -v`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add tests/test_model_pool.py main.py
git commit -m "feat: route OCR jobs through AI model pool"
```

### Task 4: Verify API and regression behavior

**Files:**
- Test: `python3 -m unittest tests/test_model_pool.py tests/test_review_api.py -v`

- [ ] **Step 1: Run the focused test suite**

Run: `python3 -m unittest tests/test_model_pool.py tests/test_review_api.py -v`
Expected: all tests `OK`

- [ ] **Step 2: Run a syntax-level verification**

Run: `python3 -m py_compile main.py tests/test_model_pool.py tests/test_review_api.py`
Expected: no output

- [ ] **Step 3: Commit**

```bash
git add main.py tests/test_model_pool.py tests/test_review_api.py docs/superpowers/plans/2026-08-31-ia-ocr-pool.md
git commit -m "feat: implement AI OCR review queue"
```
