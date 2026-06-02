"""Unit tests for observation_purifier — known inputs → expected outputs."""

import pytest
from plugins.ecoseek.observation_purifier import purify_output, purify_hermes_response


class TestPurifyOutput:
    """Test raw CLI output purification."""

    def test_strips_ansi_codes(self):
        raw = "\x1b[32mSuccess\x1b[0m: job submitted"
        result = purify_output(raw, "job_submit")
        assert "\x1b[" not in result
        assert "Success" in result or "job" in result

    def test_strips_progress_bars(self):
        raw = "Progress: [################] 100%\nSubmitted batch job 99999"
        result = purify_output(raw, "job_submit")
        assert "####" not in result
        assert "99999" in result

    def test_strips_git_noise(self):
        raw = (
            "remote: Enumerating objects: 15, done.\n"
            "remote: Counting objects: 100% (15/15), done.\n"
            "remote: Compressing objects: 100% (8/8), done.\n"
            "Already up to date.\n"
            "RUNNING 5 jobs"
        )
        result = purify_output(raw, "status_check")
        assert "Enumerating" not in result
        assert "RUNNING" in result

    def test_extracts_job_id_on_submit(self):
        raw = (
            "\x1b[32mProgress: ######### 90%\n"
            "Submitted batch job 12345\n"
            "Remote: counting objects..."
        )
        result = purify_output(raw, "job_submit")
        assert "12345" in result

    def test_preserves_error_info(self):
        raw = "Error: libRlapack.so: cannot open shared object file"
        result = purify_output(raw, "status_check")
        assert "libRlapack" in result

    def test_strips_r_compilation_noise(self):
        raw = (
            "g++ -std=gnu++17 -I/usr/share/R/include -DNDEBUG -fpic -O2 -c foo.cpp\n"
            "installing to /usr/local/lib/R/site-library/xsdm/libs\n"
            "** R\n** byte-compile and target\n"
            "* DONE (xsdm)\n"
            "library(xsdm) loaded successfully"
        )
        result = purify_output(raw, "build")
        assert "g++ -std" not in result
        assert "DONE" in result or "loaded" in result

    def test_empty_input(self):
        result = purify_output("", "status_check")
        assert result == ""

    def test_unknown_task_type(self):
        raw = "some output here"
        result = purify_output(raw, "unknown_type")
        assert "some output" in result


class TestPurifyHermesResponse:
    """Test LLM prose response purification."""

    def test_strips_preamble(self):
        raw = "Sure! Here's the output you requested:\n\nRUNNING 5 jobs\nPENDING 2 jobs"
        result = purify_hermes_response(raw, "status_check")
        assert "Sure" not in result
        assert "RUNNING" in result

    def test_strips_outro(self):
        raw = "RUNNING 3 jobs\n\nLet me know if you need anything else!"
        result = purify_hermes_response(raw, "status_check")
        assert "Let me know" not in result
        assert "RUNNING" in result

    def test_preserves_structured_data(self):
        raw = '{"job_id": "12345", "status": "RUNNING"}'
        result = purify_hermes_response(raw, "job_submit")
        assert "12345" in result
        assert "RUNNING" in result

    def test_empty_input(self):
        result = purify_hermes_response("", "status_check")
        assert result == ""
