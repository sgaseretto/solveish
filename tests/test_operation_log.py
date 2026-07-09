import logging

from dialeng.services.operation_log import elapsed_ms, log_operation, new_operation_id


def test_new_operation_id_uses_prefix():
    op_id = new_operation_id("cell-run")

    assert op_id.startswith("cell-run-")
    assert len(op_id) > len("cell-run-")


def test_log_operation_renders_stable_key_values(caplog):
    logger = logging.getLogger("dialeng.tests.operation_log")

    with caplog.at_level(logging.INFO, logger="dialeng.tests.operation_log"):
        op_id = log_operation(
            logger,
            "cell_source_commit",
            op_id="op-123",
            notebook="demo",
            cell="abc",
            duration_ms=12.34,
            detail="has spaces",
        )

    assert op_id == "op-123"
    assert "[operation] event=cell_source_commit" in caplog.text
    assert "op_id=op-123" in caplog.text
    assert "notebook=demo" in caplog.text
    assert "cell=abc" in caplog.text
    assert "duration_ms=12.3" in caplog.text
    assert "detail='has spaces'" in caplog.text


def test_elapsed_ms_is_non_negative():
    import time

    start = time.perf_counter()
    assert elapsed_ms(start) >= 0
