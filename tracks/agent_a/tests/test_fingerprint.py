from pathlib import Path
import tempfile
import unittest

from tracks.agent_a.fingerprint import SOURCE_FILES, fingerprint_dataset


class FingerprintTest(unittest.TestCase):
    @staticmethod
    def populate(path: Path, changed: bool = False) -> None:
        path.mkdir()
        for index, name in enumerate(SOURCE_FILES):
            content = f"file-{index}\n"
            if changed and index == 0:
                content += "changed\n"
            (path / name).write_text(content)

    def test_fingerprint_ignores_path_but_changes_with_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, copy, changed = root / "first", root / "copy", root / "changed"
            self.populate(first)
            self.populate(copy)
            self.populate(changed, changed=True)
            a = fingerprint_dataset(first)["dataset_fingerprint"]
            b = fingerprint_dataset(copy)["dataset_fingerprint"]
            c = fingerprint_dataset(changed)["dataset_fingerprint"]
            self.assertEqual(a, b)
            self.assertNotEqual(a, c)


if __name__ == "__main__":
    unittest.main()
