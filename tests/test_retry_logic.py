"""Tests for _is_retryable_error and _call_with_retry."""

import pytest
import transaction_web_app as app


# ── _is_retryable_error ────────────────────────────────────────────────────

def test_retryable_on_429():
    assert app._is_retryable_error(Exception("HTTP 429 rate limit exceeded"))

def test_retryable_on_503():
    assert app._is_retryable_error(Exception("503 service unavailable"))

def test_retryable_on_timeout():
    assert app._is_retryable_error(Exception("Request timeout after 30s"))

def test_not_retryable_on_auth_error():
    assert not app._is_retryable_error(Exception("401 Unauthorized: invalid API key"))

def test_not_retryable_on_bad_request():
    assert not app._is_retryable_error(Exception("400 Bad Request"))


# ── _call_with_retry ───────────────────────────────────────────────────────

def test_succeeds_on_first_attempt():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    result = app._call_with_retry(fn, max_attempts=3, base_delay=0)
    assert result == "ok"
    assert len(calls) == 1


def test_retries_on_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    attempts = []

    def fn():
        attempts.append(1)
        if len(attempts) < 3:
            raise Exception("429 Too Many Requests")
        return "success"

    result = app._call_with_retry(fn, max_attempts=3, base_delay=0)
    assert result == "success"
    assert len(attempts) == 3


def test_raises_after_max_attempts_exhausted(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)

    def fn():
        raise Exception("503 service unavailable")

    with pytest.raises(Exception, match="503"):
        app._call_with_retry(fn, max_attempts=3, base_delay=0)


def test_does_not_retry_non_retryable_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    attempts = []

    def fn():
        attempts.append(1)
        raise ValueError("401 Unauthorized")

    with pytest.raises(ValueError):
        app._call_with_retry(fn, max_attempts=3, base_delay=0)

    assert len(attempts) == 1  # no retry for auth errors
