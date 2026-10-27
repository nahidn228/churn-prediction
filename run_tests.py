"""
Convenience entrypoint script to run the test suite.
Can be executed as: python run_tests.py
"""
import sys
import pytest

if __name__ == "__main__":
    print("=" * 60)
    print("  Running Customer Churn & Segmentation Test Suite  ")
    print("=" * 60)
    exit_code = pytest.main(["-v", "tests"])
    sys.exit(exit_code)
