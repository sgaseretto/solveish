from pathlib import Path

import pytest

from dialeng.services.llm.providers.claudette_provider import ClaudetteProvider


def test_claudette_accepts_only_bedrock():
    assert ClaudetteProvider()._backend == "bedrock"
    assert ClaudetteProvider(backend="bedrock")._backend == "bedrock"
    with pytest.raises(ValueError, match="Unsupported Claudette backend"):
        ClaudetteProvider(backend="unsupported")


def test_removed_provider_has_no_repository_references():
    root = Path(__file__).parents[1]
    forbidden = (
        "claude" + "_agent_sdk",
        "claude" + "-agent-sdk",
        "claude" + "_code_subscription",
        "claudette:" + "anthropic_api",
        "ANTHROPIC" + "_API_KEY",
        "test_claude" + "_agent_query",
    )
    roots = [
        root / "dialeng",
        root / "tests",
        root / "docs",
        root / "notebooks",
    ]
    files = [root / "README.md", root / "CHANGELOG.md", root / "pyproject.toml"]
    for search_root in roots:
        files.extend(
            path for path in search_root.rglob("*")
            if path.is_file() and path.suffix in {".py", ".md", ".toml", ".json", ".ipynb"}
        )

    violations = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden:
            if token in text:
                violations.append(f"{path.relative_to(root)}: {token}")
    assert not violations, "Removed provider references remain:\n" + "\n".join(violations)
