import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from chess_intuition.data.preprocess import preprocess_csv


class PreprocessTests(unittest.TestCase):
    def test_first_moves_and_legal_masks_are_saved_by_position_group(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            csv_path = root / "puzzles.csv"
            output = root / "processed"
            fen_a = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            fen_b = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 7 22"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["FEN", "Moves", "Rating"])
                writer.writeheader()
                writer.writerow({"FEN": fen_a, "Moves": "e2e4 e7e5", "Rating": 1200})
                writer.writerow({"FEN": fen_b, "Moves": "d2d4 d7d5", "Rating": 1300})

            manifest = preprocess_csv(csv_path, output, shard_size=10)
            self.assertEqual(manifest["rows_read"], 2)
            self.assertEqual(manifest["rows_kept"], 2)
            self.assertEqual(sum(manifest["split_counts"].values()), 2)
            self.assertEqual(len(list((output / "train").glob("part-*.npz"))) +
                             len(list((output / "validation").glob("part-*.npz"))) +
                             len(list((output / "test").glob("part-*.npz"))), 1)

            split = next(name for name in ("train", "validation", "test") if list((output / name).glob("part-*.npz")))
            shard_path = next((output / split).glob("part-*.npz"))
            with np.load(shard_path, allow_pickle=False) as shard:
                self.assertEqual(len(shard["policy"]), 2)
                offsets = shard["legal_offsets"]
                legal = shard["legal_actions"]
                for index, target in enumerate(shard["policy"]):
                    candidates = legal[int(offsets[index]) : int(offsets[index + 1])]
                    self.assertIn(int(target), candidates.tolist())
            saved_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(saved_manifest["value_labels"])

    def test_nonempty_output_fails_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "unused.csv"
            source.write_text("FEN,Moves,Rating\n", encoding="utf-8")
            output = root / "processed"
            output.mkdir()
            (output / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                preprocess_csv(source, output)
            self.assertTrue((output / "keep.txt").exists())


if __name__ == "__main__":
    unittest.main()

