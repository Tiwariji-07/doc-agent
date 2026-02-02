#!/usr/bin/env python3
"""
Test script to diagnose Academy MCP connection issues.

Usage:
    python test_mcp_connection.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from src.config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def test_mcp_connection():
    """Test connection to Academy MCP server."""
    settings = get_settings()
    mcp_url = settings.academy_mcp_url

    print("\n" + "=" * 70)
    print("Academy MCP Connection Test")
    print("=" * 70)

    if not mcp_url:
        print("❌ ACADEMY_MCP_URL is not configured in .env")
        print("   Please set: ACADEMY_MCP_URL=https://dev-academyservices.wavemaker.com/mcp")
        return

    print(f"\n✓ MCP URL configured: {mcp_url}\n")

    # Step 1: Test basic HTTP connectivity
    print("Step 1: Testing basic HTTP connectivity...")
    try:
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(mcp_url)
            print(f"   ✓ HTTP {response.status_code}: {response.reason_phrase}")
            print(f"   Response length: {len(response.text)} bytes")
            if response.status_code != 200:
                print(f"   Response preview: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ HTTP request failed: {type(e).__name__}: {e}")
        return

    # Step 2: Initialize MCP client
    print("\nStep 2: Initializing MCP streamable HTTP client...")
    try:
        context = streamablehttp_client(mcp_url)
        read_stream, write_stream, _ = await context.__aenter__()
        print("   ✓ Streamable HTTP client created")
    except Exception as e:
        print(f"   ❌ Failed to create client: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return

    # Step 3: Create MCP session
    print("\nStep 3: Creating MCP session...")
    try:
        session = ClientSession(read_stream, write_stream)
        await session.__aenter__()
        print("   ✓ MCP session created")
    except Exception as e:
        print(f"   ❌ Failed to create session: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        await context.__aexit__(None, None, None)
        return

    # Step 4: Initialize session
    print("\nStep 4: Initializing MCP session (protocol handshake)...")
    try:
        init_result = await session.initialize()
        print("   ✓ Session initialized")
        print(f"   Server info: {init_result}")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        await session.__aexit__(None, None, None)
        await context.__aexit__(None, None, None)
        return

    # Step 5: List available tools
    print("\nStep 5: Listing available tools...")
    try:
        tools_result = await session.list_tools()
        print(f"   ✓ Found {len(tools_result.tools)} tools:")
        for tool in tools_result.tools:
            print(f"      - {tool.name}: {tool.description}")
    except Exception as e:
        print(f"   ❌ Failed to list tools: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    # Step 6: Call the wm-academy-semantic-search tool
    print("\nStep 6: Calling wm-academy-semantic-search tool...")
    try:
        result = await session.call_tool(
            name="wm-academy-semantic-search",
            arguments={
                "query": "REST API",
                "limit": 3,
            },
        )
        print("   ✓ Tool call successful")
        print(f"   Response type: {type(result)}")
        print(f"   Content items: {len(result.content)}")

        if result.content:
            response_text = result.content[0].text
            print(f"\n   Raw response preview ({len(response_text)} chars):")
            print(f"   {response_text[:300]}...")

            # Parse JSON
            data = json.loads(response_text)
            videos = data.get("body", [])
            print(f"\n   ✓ Found {len(videos)} videos:")
            for video in videos[:3]:
                print(f"      - [{video['code']}] {video['title']}")
                print(f"        {video['link']}")

    except Exception as e:
        print(f"   ❌ Tool call failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    # Cleanup
    print("\nStep 7: Cleaning up...")
    try:
        await session.__aexit__(None, None, None)
        await context.__aexit__(None, None, None)
        print("   ✓ Resources cleaned up")
    except Exception as e:
        print(f"   ⚠️  Cleanup warning: {e}")

    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(test_mcp_connection())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
