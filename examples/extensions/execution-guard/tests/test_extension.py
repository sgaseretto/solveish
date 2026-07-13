from extension import setup


def test_setup_is_async():
    import inspect
    assert inspect.iscoroutinefunction(setup)
