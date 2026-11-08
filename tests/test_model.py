import unittest
from collector import parse_draw
from model import probability_vectors,walkforward

class TestNyx(unittest.TestCase):
    def test_parse(self):
        value='Draw No: 6188\nDate: Wednesday May 13, 2026\nG883017\n'
        self.assertEqual(parse_draw(value,6188),dict(draw_number=6188,draw_date='2026-05-13',letter='G',digits='883017'))
        self.assertIsNone(parse_draw(value,6189))
    def test_model(self):
        rows=[dict(draw_number=i,letter='A',digits='012345') for i in range(35)]
        vec=probability_vectors(rows,'uniform')
        self.assertEqual(len(vec),7)
        self.assertAlmostEqual(sum(vec[0]),1)
        chosen,scores=walkforward(rows)
        self.assertIn(chosen,scores)
if __name__=='__main__':unittest.main()
