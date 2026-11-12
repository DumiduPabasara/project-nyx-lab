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
        self.assertEqual(chosen,'uniform')
        self.assertEqual(scores['uniform']['validation_cases'],7)
        self.assertEqual(scores['uniform']['test_cases'],8)
        self.assertIn('test_mean_brier_score',scores['uniform'])
        self.assertLessEqual(
            scores['uniform']['test_log_loss_95ci_low'],
            scores['uniform']['test_mean_log_loss'],
        )
        self.assertGreaterEqual(
            scores['uniform']['test_log_loss_95ci_high'],
            scores['uniform']['test_mean_log_loss'],
        )

    def test_holdout_does_not_affect_model_selection(self):
        rows=[
            dict(draw_number=i,letter='A' if i%2==0 else 'B',
                 digits='012345' if i%2==0 else '987654')
            for i in range(100)
        ]
        changed=[dict(row) for row in rows]
        for row in changed[60:]:
            row['letter']='Z'
            row['digits']='999999'

        chosen,scores=walkforward(rows)
        changed_chosen,changed_scores=walkforward(changed)

        self.assertEqual(chosen,changed_chosen)
        self.assertEqual(
            scores['all_history']['validation_mean_log_loss'],
            changed_scores['all_history']['validation_mean_log_loss'],
        )
        self.assertNotEqual(
            scores['all_history']['test_mean_log_loss'],
            changed_scores['all_history']['test_mean_log_loss'],
        )
if __name__=='__main__':unittest.main()
