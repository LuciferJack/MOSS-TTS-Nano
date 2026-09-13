import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from finetuning.sft import apply_train_schedule


class TrainScheduleTest(unittest.TestCase):
    def apply(self, records, schedule):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.json"
            path.write_text(json.dumps(schedule), encoding="utf-8")
            return apply_train_schedule(records, str(path))

    def test_exact_schedule_reorders_records(self):
        records = [{"id": "hydrate-a"}, {"id": "replay"}, {"id": "hydrate-b"}]
        ordered, resolved = self.apply(records, ["hydrate-a", "replay", "hydrate-b"])
        self.assertEqual([row["id"] for row in ordered], resolved)

    def test_missing_unknown_and_duplicate_ids_fail_closed(self):
        records = [{"id": "a"}, {"id": "b"}]
        for schedule, message in ((["a"], "exact permutation"),
                                  (["a", "x"], "exact permutation"),
                                  (["a", "a"], "duplicate")):
            with self.subTest(schedule=schedule):
                with self.assertRaisesRegex(ValueError, message):
                    self.apply(records, schedule)

    def test_duplicate_record_ids_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "records contain duplicate"):
            self.apply([{"id": "a"}, {"id": "a"}], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
