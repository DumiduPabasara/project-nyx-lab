import unittest

from collector import parse_draw


class CollectorTests(unittest.TestCase):
    def test_rejects_letter_outside_alphabet(self):
        page = 'Draw No: 6188\nDate: Wednesday May 13, 2026\nI883017\n'
        self.assertIsNone(parse_draw(page, 6188))

    def test_accepts_valid_letter(self):
        page = 'Draw No: 6188\nDate: Wednesday May 13, 2026\nJ883017\n'
        self.assertEqual(parse_draw(page, 6188)['letter'], 'J')


if __name__ == '__main__':
    unittest.main()