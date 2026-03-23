"""Tests for Dialeng console log configuration."""

import logging

from dialeng.logging_config import DialengAccessFilter, build_log_config


def _access_record(path: str, status_code: int = 200) -> logging.LogRecord:
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:8000", "GET", path, "1.1", status_code),
        exc_info=None,
    )


def test_access_filter_suppresses_snapshot_polling():
    assert DialengAccessFilter().filter(_access_record("/dialeng/demo/kernel/snapshot")) is False


def test_access_filter_keeps_errors_visible():
    assert DialengAccessFilter().filter(_access_record("/dialeng/demo/kernel/snapshot", status_code=404)) is True


def test_build_log_config_registers_dialeng_logger_and_access_filter():
    config = build_log_config()

    assert config["handlers"]["access"]["filters"] == ["dialeng_access"]
    assert config["loggers"]["dialeng"]["handlers"] == ["dialeng"]
    assert config["formatters"]["dialeng"]["fmt"] == "%(levelprefix)s [%(name)s] %(message)s"
