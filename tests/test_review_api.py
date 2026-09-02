import asyncio
import json
import unittest
from unittest.mock import patch

import main


class ReviewApiTests(unittest.TestCase):
    @patch("main.fetch_job_row")
    @patch("main.update_job_result_payload")
    def test_patch_review_saves_review_draft(self, update_job_result_payload, fetch_job_row):
        fetch_job_row.return_value = {
            "id": 1,
            "status": "review_required",
            "result": json.dumps({"reviewDraft": {"nome": "A"}}),
        }

        response = asyncio.run(main.save_review(1, {"reviewDraft": {"nome": "B"}}))

        self.assertEqual(response["jobId"], 1)
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

        response = asyncio.run(main.confirm_job(1))

        self.assertEqual(response.status_code, 400)
        mark_job_confirmed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
