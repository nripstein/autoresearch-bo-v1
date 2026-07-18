import unittest

from propose_and_run import matches_requested


class ProposalValidationTests(unittest.TestCase):
    def test_requested_values_must_match_reported_values(self):
        result = {"lr": 0.04, "weight_decay": 0.2, "val_bpb": 1.0}
        self.assertTrue(matches_requested(result, 0.04, 0.2))
        self.assertTrue(matches_requested(result, 0.04000000000001, 0.2))
        self.assertFalse(matches_requested(result, 0.041, 0.2))
        self.assertFalse(matches_requested(result, 0.04, 0.21))
