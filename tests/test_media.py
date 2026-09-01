import tempfile
import unittest
from pathlib import Path

from frameledger.media import MediaError, validate_video_path


class MediaSafetyTests(unittest.TestCase):
    def test_rejects_incomplete_part_before_processing(self):
        with self.assertRaisesRegex(MediaError, "Incomplete download"):
            validate_video_path("/tmp/example.mp4.part")

    def test_rejects_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(MediaError, "one video file"):
                validate_video_path(Path(directory))

    def test_rejects_unsupported_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text("not video", encoding="utf-8")
            with self.assertRaisesRegex(MediaError, "Unsupported video extension"):
                validate_video_path(path)


if __name__ == "__main__":
    unittest.main()
