#!/usr/bin/env python3
"""
CLI script to index WaveMaker documentation.

Usage:
    python scripts/index_docs.py
    python scripts/index_docs.py --force
    python scripts/index_docs.py --local /path/to/docs
    python scripts/index_docs.py --branch release-11
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexer.indexer import run_indexer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    parser = argparse.ArgumentParser(
        description="Index WaveMaker documentation into Qdrant",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Git branch to index (default: from .env)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full reindex (deletes existing collection)",
    )
    parser.add_argument(
        "--local",
        type=str,
        default=None,
        help="Local path to docs instead of cloning from GitHub",
    )

    args = parser.parse_args()

    print("\n🚀 WaveMaker Docs Indexer")
    print("=" * 40)

    # Check for .env file
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        print("\n❌ Error: .env file not found!")
        print("   Copy .env.example to .env and configure it.")
        sys.exit(1)

    # Run indexer
    result = asyncio.run(
        run_indexer(
            branch=args.branch,
            force=args.force,
            local_path=args.local,
        )
    )

    # Exit with appropriate code
    if result["status"] == "failed":
        sys.exit(1)
    elif result["errors"]:
        sys.exit(0)  # Partial success
    else:
        sys.exit(0)  # Full success


if __name__ == "__main__":
    main()
