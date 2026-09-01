import unittest

from frameledger.timecode import format_timecode, parse_timecode


class TimecodeTests(unittest.TestCase):
    def test_parse_supported_forms(self):
        self.assertEqual(parse_timecode("75.5"), 75.5)
        self.assertEqual(parse_timecode("01:15.5"), 75.5)
        self.assertEqual(parse_timecode("01:02:03.250"), 3723.25)

    def test_reject_negative_and_invalid(self):
        for value in ("", "-1", "1:2:3:4", float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_timecode(value)

    def test_format_rounds_to_milliseconds(self):
        self.assertEqual(format_timecode(3723.2496), "01:02:03.250")
        self.assertEqual(format_timecode(65.2, milliseconds=False), "00:01:05")


if __name__ == "__main__":
    unittest.main()
