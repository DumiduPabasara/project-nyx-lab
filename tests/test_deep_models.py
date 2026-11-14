import unittest

from deep_models import MIN_HISTORY, available, encode_rows, evaluate_deep_models
from model import ALPHABET


class DeepModelTests(unittest.TestCase):
    def test_neural_models_wait_for_enough_history(self):
        rows = [
            {"draw_number": i + 1, "letter": ALPHABET[i % len(ALPHABET)],
             "digits": f"{i % 10}{(i + 1) % 10}{(i + 2) % 10}{(i + 3) % 10}{(i + 4) % 10}{(i + 5) % 10}"}
            for i in range(100)
        ]
        result = evaluate_deep_models(rows)
        self.assertFalse(result["available"])
        self.assertIn(str(MIN_HISTORY), result["reason"])

    @unittest.skipUnless(available(), "optional PyTorch/NumPy dependency not installed")
    def test_one_hot_encoding_has_85_features_and_seven_active_symbols(self):
        rows = [
            {"draw_number": 1, "letter": "A", "digits": "012345"},
            {"draw_number": 2, "letter": "Z", "digits": "987654"},
        ]
        encoded = encode_rows(rows)
        self.assertEqual(encoded.shape, (2, 85))
        self.assertEqual(float(encoded[0].sum()), 7.0)
        self.assertEqual(float(encoded[1].sum()), 7.0)


if __name__ == "__main__":
    unittest.main()
