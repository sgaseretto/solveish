#!/usr/bin/env python
"""
Test script for verifying stateless behavior in LLM queries.

This test simulates the John→Mark scenario:
1. Send "Hello! My name is John" → Claude responds
2. Update context to change "John" → "Mark"
3. Ask "Can you remind me my name?" → Claude should say "Mark" (not "John")

Usage:
    uv run python test_stateless_dialoghelper.py

The test verifies that the LLM doesn't remember previous session state,
ensuring the notebook cells are the sole source of truth.
"""
import asyncio
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# IMPORTANT: Detect credentials BEFORE entering async context
# This is needed because the credential probe creates its own event loop
print("Pre-loading credentials (before async context)...")
from services.credential_service import detect_credentials
_creds = detect_credentials()
print(f"Detected: provider={_creds.provider}, backend={_creds.backend}")


async def test_stateless_sdk_direct():
    """Test stateless behavior using SDK directly."""
    print("\n" + "=" * 60)
    print("Testing STATELESS behavior with claude-agent-sdk direct")
    print("=" * 60)

    # Create LLMService - credentials already detected above
    from services.llm_service import LLMService
    service = LLMService()
    print(f"Using provider: {_creds.provider}, backend: {_creds.backend}")

    # Query 1: "Hello! My name is John"
    print("\n[Step 1] Sending: 'Hello! My name is John'")
    context1 = []  # No context for first message
    response1 = []

    async for item in service.stream_response(
        prompt="Hello! My name is John",
        context_messages=context1,
        mode="standard",
        model="claude-sonnet-3-7",
        use_thinking=False
    ):
        if item["type"] == "chunk":
            response1.append(item["content"])
        elif item["type"] == "error":
            print(f"ERROR: {item['content']}")
            return False

    response1_text = "".join(response1)
    print(f"Response 1: {response1_text[:200]}...")

    # Verify response 1 mentions John
    if "john" not in response1_text.lower():
        print("WARNING: Response 1 doesn't mention John")

    # Query 2: Simulate EDITING the cell to say "Mark" instead of "John"
    # The context now shows "Mark" not "John"
    print("\n[Step 2] Context updated: 'Hello! My name is Mark' (simulating cell edit)")
    print("[Step 3] Sending: 'Can you remind me my name?'")

    context2 = [
        {"role": "user", "content": "Hello! My name is Mark"},
        {"role": "assistant", "content": "Hello Mark! Nice to meet you."},
    ]

    response2 = []
    async for item in service.stream_response(
        prompt="Can you remind me my name?",
        context_messages=context2,
        mode="standard",
        model="claude-sonnet-3-7",
        use_thinking=False
    ):
        if item["type"] == "chunk":
            response2.append(item["content"])
        elif item["type"] == "error":
            print(f"ERROR: {item['content']}")
            return False

    response2_text = "".join(response2)
    print(f"Response 2: {response2_text}")

    # Check results
    print("\n" + "-" * 60)
    print("TEST RESULTS:")
    print("-" * 60)

    response2_lower = response2_text.lower()
    has_mark = "mark" in response2_lower
    has_john = "john" in response2_lower

    if has_mark and not has_john:
        print("✅ PASS: Response correctly says 'Mark' and doesn't mention 'John'")
        return True
    elif has_mark and has_john:
        print("❌ FAIL: Response mentions BOTH 'Mark' AND 'John' - session contamination detected!")
        print(f"   Full response: {response2_text}")
        return False
    elif has_john and not has_mark:
        print("❌ FAIL: Response says 'John' instead of 'Mark' - session contamination!")
        print(f"   Full response: {response2_text}")
        return False
    else:
        print("⚠️ UNCERTAIN: Response doesn't clearly mention either name")
        print(f"   Full response: {response2_text}")
        return False


async def test_compare_providers():
    """Compare SDK direct vs claudette-agent wrapper."""
    print("\n" + "=" * 60)
    print("Comparing SDK direct vs claudette-agent wrapper")
    print("=" * 60)

    from services.dialeng_config import load_config, reset_config_cache

    # Test with SDK direct
    print("\n--- Testing with use_sdk_direct=True ---")
    reset_config_cache()
    # Manually set config (or modify dialeng_config.json)
    result_sdk = await test_stateless_sdk_direct()

    print("\n" + "=" * 60)
    print(f"SDK Direct Result: {'PASS' if result_sdk else 'FAIL'}")
    print("=" * 60)

    return result_sdk


async def main():
    """Run all tests."""
    print("=" * 60)
    print("STATELESS QUERY TEST SUITE")
    print("=" * 60)
    print("This test verifies that editing notebook cells is reflected")
    print("in subsequent LLM queries without session contamination.")
    print()
    print("Test scenario: John → Mark name change")
    print("1. Send 'Hello! My name is John'")
    print("2. Edit context to say 'Mark' instead of 'John'")
    print("3. Ask 'Can you remind me my name?'")
    print("4. Verify response says 'Mark', NOT 'John'")

    success = await test_compare_providers()

    print("\n" + "=" * 60)
    print("FINAL RESULT:")
    if success:
        print("✅ All tests PASSED - Stateless queries working correctly")
        sys.exit(0)
    else:
        print("❌ Tests FAILED - Session contamination detected")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
