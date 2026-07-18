import unittest

from fingerprint import get_fingerprint


class FingerprintTests(unittest.TestCase):
    def test_fingerprint_is_short_and_stable(self):
        self.assertEqual(get_fingerprint(), get_fingerprint())
        self.assertRegex(get_fingerprint(), r"^[0-9a-f]{10}$")
