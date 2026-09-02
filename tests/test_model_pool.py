import unittest
from urllib.error import HTTPError
from unittest.mock import patch

import main


class ModelPoolTests(unittest.TestCase):
    def test_default_models_are_api_ready_slugs(self):
        self.assertIn(":", main.OPENROUTER_MODELS[0])
        self.assertTrue(all("/" in model for model in main.OPENROUTER_MODELS))

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

    @patch("main.send_discord_alert")
    def test_http_429_puts_slot_in_cooldown_and_alerts(self, send_discord_alert):
        slot = main.ModelSlot(model="a", cooldown_until=0.0, consecutive_failures=0, last_error="")
        err = HTTPError("http://x", 429, "rate limit", hdrs=None, fp=None)

        with self.assertRaises(HTTPError):
            main.handle_model_exception(slot, err, now=100.0)

        self.assertEqual(slot.last_error, "http_429")
        send_discord_alert.assert_called_once()


if __name__ == "__main__":
    unittest.main()
