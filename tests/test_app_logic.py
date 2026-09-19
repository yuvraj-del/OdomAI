import unittest

from backend.app import apply_luxury_price_adjustment, compute_confidence


class AppLogicRegressionTests(unittest.TestCase):
    def test_luxury_brand_normalization_for_mercedes_and_alfa(self):
        self.assertAlmostEqual(
            apply_luxury_price_adjustment(10000, 'mercedes', 'c class'),
            7200,
            places=2,
        )
        self.assertAlmostEqual(
            apply_luxury_price_adjustment(10000, 'mercedes-benz', 'c class'),
            7200,
            places=2,
        )
        self.assertAlmostEqual(
            apply_luxury_price_adjustment(10000, 'alfa romeo', 'giulia'),
            8000,
            places=2,
        )
        self.assertAlmostEqual(
            apply_luxury_price_adjustment(10000, 'alfa-romeo', 'giulia'),
            8000,
            places=2,
        )

    def test_rs_match_is_whole_token_only(self):
        self.assertEqual(apply_luxury_price_adjustment(10000, 'nissan', 'versa'), 10000.0)
        self.assertEqual(apply_luxury_price_adjustment(10000, 'nissan', 'traverse'), 10000.0)

    def test_confidence_prefers_mainstream_midrange(self):
        mainstream = compute_confidence(2012, 110000)
        new_low_mileage = compute_confidence(2024, 1000)
        old_high_mileage = compute_confidence(1996, 300000)

        self.assertGreater(mainstream, new_low_mileage)
        self.assertGreater(mainstream, old_high_mileage)
        self.assertGreaterEqual(mainstream, 80)
        self.assertLessEqual(mainstream, 96)


if __name__ == '__main__':
    unittest.main()
