import csv
import tempfile
import unittest
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess
from unittest.mock import patch

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

    @patch("run_logging.subprocess.run")
    def test_append_records_current_branch(self, mock_run):
        mock_run.return_value = CompletedProcess(
            args=["git", "rev-parse", "--abbrev-ref", "HEAD"], returncode=0,
            stdout="autoresearch/mar5\n",
        )
        with tempfile.TemporaryDirectory() as directory:
            row = append_run(
                fingerprint="abc", source="agent", wall_clock_seconds=1.2,
                result={"lr": 0.04, "weight_decay": 0.2, "val_bpb": 1.0},
                path=Path(directory) / "runs.csv",
            )
        self.assertEqual(row["branch"], "autoresearch/mar5")
        mock_run.assert_called_once_with(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True, capture_output=True, text=True,
        )

    @patch("run_logging.subprocess.run", side_effect=CalledProcessError(1, "git"))
    def test_append_logs_when_branch_lookup_fails(self, mock_run):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.csv"
            row = append_run(
                fingerprint="abc", source="agent", wall_clock_seconds=1.2,
                result={"lr": 0.04, "weight_decay": 0.2, "val_bpb": 1.0}, path=path,
            )
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(row["branch"], "")
        self.assertEqual(rows[0]["branch"], "")
        mock_run.assert_called_once()

    def test_legacy_csv_loads_with_blank_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.csv"
            path.write_text(
                "timestamp,iteration,fingerprint,lr,weight_decay,val_bpb,"
                "wall_clock_seconds,source,requested_lr,requested_weight_decay,"
                "n_prior_obs,cold_start\n"
                "2026-03-05T00:00:00Z,1,abc,0.04,0.2,1.0,1.200,agent,,,,\n"
            )
            rows = load_successful_runs(path)
        self.assertEqual(rows[0]["branch"], "")
        self.assertEqual(rows[0]["val_bpb"], 1.0)

    @patch("run_logging.subprocess.run")
    def test_append_migrates_legacy_csv_without_backfilling_branch(self, mock_run):
        mock_run.return_value = CompletedProcess(args=["git"], returncode=0, stdout="autoresearch/mar12\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.csv"
            path.write_text(
                "timestamp,iteration,fingerprint,lr,weight_decay,val_bpb,"
                "wall_clock_seconds,source,requested_lr,requested_weight_decay,"
                "n_prior_obs,cold_start\n"
                "2026-03-05T00:00:00Z,1,abc,0.04,0.2,1.0,1.200,agent,,,,\n"
            )
            append_run(
                fingerprint="abc", source="tool", wall_clock_seconds=1.3,
                result={"lr": 0.03, "weight_decay": 0.1, "val_bpb": 0.9}, path=path,
            )
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["branch"], "")
        self.assertEqual(rows[1]["branch"], "autoresearch/mar12")
        self.assertEqual(list(rows[0]), RUN_FIELDS)
