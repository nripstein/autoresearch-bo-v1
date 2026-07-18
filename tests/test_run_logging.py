import csv
import tempfile
import unittest
from pathlib import Path

from run_logging import RESULT_PREFIX, RUN_FIELDS, append_run, load_successful_runs, parse_result_line


class RunLoggingTests(unittest.TestCase):
    def test_parse_structured_result(self):
        result = parse_result_line(RESULT_PREFIX + '{"lr": 0.04, "weight_decay": 0.2, "val_bpb": 1.23}')
        self.assertEqual(result, {"lr": 0.04, "weight_decay": 0.2, "val_bpb": 1.23})
        self.assertIsNone(parse_result_line("val_bpb: 1.23"))

    def test_append_assigns_iterations_and_skips_incomplete_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.csv"
            first = append_run(fingerprint="abc", source="agent", wall_clock_seconds=1.2,
                               result=None, path=path)
            second = append_run(fingerprint="abc", source="agent", wall_clock_seconds=1.3,
                                result={"lr": 0.04, "weight_decay": 0.2, "val_bpb": 1.0}, path=path)
            self.assertEqual((first["iteration"], second["iteration"]), (1, 2))
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(list(rows[0]), RUN_FIELDS)
            self.assertEqual(len(rows), 2)
            self.assertEqual(load_successful_runs(path), [{**rows[1], "lr": 0.04, "weight_decay": 0.2, "val_bpb": 1.0}])
