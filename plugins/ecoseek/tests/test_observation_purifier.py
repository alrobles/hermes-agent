"""Unit tests for observation_purifier.py."""
import sys
sys.path.insert(0, "/home/reumanlab/hermes-agent-fork")

import unittest
from plugins.ecoseek.observation_purifier import (
    purify_output,
    purify_hermes_response,
)


class TestPurifyOutput(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(purify_output(""), "")

    def test_ansi_stripped(self):
        raw = "\x1b[32mProgress: ######### 90%\nSubmitted batch job 12345\nRemote: counting..."
        result = purify_output(raw, "job_submit")
        self.assertIn("job_id:12345", result)
        self.assertIn("status:submitted", result)
        self.assertNotIn("\x1b[32m", result)

    def test_progress_bar_stripped(self):
        raw = "########## 100%\nJob 22285076 completed"
        result = purify_output(raw)
        self.assertIn("completed", result)
        self.assertNotIn("##########", result)

    def test_git_noise_stripped(self):
        raw = "Cloning into 'repo'... 100%\nReceiving objects: 50% (10/20)\nremote: Counting objects: 100%\noutput.txt"
        result = purify_output(raw)
        self.assertIn("output.txt", result)
        self.assertNotIn("Receiving objects", result)

    def test_traceback_compressed(self):
        raw = """Traceback (most recent call last):
  File "/app/main.py", line 42, in <module>
    result = do_work()
  File "/app/main.py", line 30, in do_work
    1/0
ZeroDivisionError: division by zero"""
        result = purify_output(raw)
        self.assertIn("Traceback:", result)
        self.assertIn("ZeroDivisionError", result)
        self.assertIn("more frames", result)

    def test_status_check_extracts_job_lines(self):
        raw = """some noise
22285076 RUNNING nem-m1
22285077 PENDING test-job"""
        result = purify_output(raw, "status_check")
        self.assertIn("RUNNING", result)
        self.assertIn("PENDING", result)
        self.assertNotIn("some noise", result)  # noise line without job ID stripped

    def test_job_submit_extracts_id(self):
        raw = "Some noise\nSubmitted batch job 22285076\nmore noise"
        result = purify_output(raw, "job_submit")
        self.assertEqual(result, "job_id:22285076\nstatus:submitted")

    def test_truncation(self):
        raw = "x" * 3000
        result = purify_output(raw)
        self.assertLess(len(result), 2500)
        self.assertIn("[truncated]", result)


class TestPurifyHermesResponse(unittest.TestCase):
    def test_json_preserved(self):
        resp = '{"status": "ok", "job_id": "12345"}'
        self.assertEqual(purify_hermes_response(resp), resp)

    def test_preambles_removed(self):
        resp = "Sure, I'll help you with that. Here's the job status: RUNNING"
        result = purify_hermes_response(resp)
        self.assertNotIn("Sure, I'll help you", result)
        self.assertIn("RUNNING", result)

    def test_outros_removed(self):
        resp = "Job is running\n\nLet me know if you need anything else."
        result = purify_hermes_response(resp)
        self.assertNotIn("Let me know", result)

    def test_empty(self):
        self.assertEqual(purify_hermes_response(""), "")


if __name__ == "__main__":
    unittest.main()
