import unittest
from unittest import mock
from backend import app as app_module
from backend.app import (
    CONFIDENCE_BOOST,
    CONFIDENCE_LIMITS,
    CONFIDENCE_MAX,
    MODEL_LISTING_COUNTS,
    RARE_MODEL_MAX_LISTINGS,
    apply_luxury_price_adjustment,
    compute_confidence,
)


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

    def test_confidence_falls_with_age_and_mileage(self):
        self.assertGreater(compute_confidence(2020, 30000), compute_confidence(2012, 120000))
        self.assertGreater(compute_confidence(2012, 120000), compute_confidence(1998, 200000))
        self.assertGreater(compute_confidence(2015, 40000), compute_confidence(2015, 200000))

    def test_confidence_is_lower_for_cars_newer_than_the_training_data(self):
        self.assertLess(compute_confidence(2026, 5000), compute_confidence(2020, 5000))
        self.assertLess(compute_confidence(2026, 5000), compute_confidence(2024, 5000))

    def test_confidence_is_lower_for_rare_models(self):
        common = next(m for m, n in MODEL_LISTING_COUNTS.items() if n >= 500)
        rare = next(m for m, n in MODEL_LISTING_COUNTS.items() if n < RARE_MODEL_MAX_LISTINGS)
        self.assertLess(compute_confidence(2012, 100000, rare), compute_confidence(2012, 100000, common))

    def test_confidence_stays_within_limits(self):
        for year, miles in [(1995, 300000), (2026, 1000), (2012, 100000)]:
            rating = compute_confidence(year, miles)
            self.assertTrue(CONFIDENCE_LIMITS[0] + CONFIDENCE_BOOST <= rating <= CONFIDENCE_MAX)

    def test_confidence_rating_adds_the_boost(self):
        # 2015 / 80,000 mi: base 87 - 1.7*11 - 0.15*8 = 67.1 -> 67, so the rating is 67 + boost.
        self.assertEqual(compute_confidence(2015, 80000), 67 + CONFIDENCE_BOOST)
        # 2020 / 30,000 mi: base 87 - 1.7*6 - 0.15*3 = 76.35 -> 76, so the rating is 91.
        self.assertEqual(compute_confidence(2020, 30000), 91)

    def test_confidence_rating_is_capped(self):
        # With today's constants no car reaches the cap (the best base score is about 78), so raise the boost to test it.
        with mock.patch.object(app_module, "CONFIDENCE_BOOST", 40):
            self.assertEqual(compute_confidence(2020, 30000), CONFIDENCE_MAX)

    def test_progressive_price_reducer_hits_target_ranges(self):
        self.assertAlmostEqual(apply_luxury_price_adjustment(15000, 'toyota', 'corolla'), 15000.0, places=2)
        self.assertGreater(apply_luxury_price_adjustment(15001, 'toyota', 'corolla'), 15000.0)
        self.assertAlmostEqual(apply_luxury_price_adjustment(20000, 'toyota', 'corolla'), 17000.0, places=1)
        self.assertAlmostEqual(apply_luxury_price_adjustment(40000, 'toyota', 'corolla'), 25600.0, places=1)


if __name__ == '__main__':
    unittest.main()
