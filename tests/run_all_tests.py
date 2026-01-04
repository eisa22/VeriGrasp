#!/usr/bin/env python3
# tests/run_all_tests.py

"""
Test Runner für die Vision-to-Grasp Pipeline.

Usage:
    python tests/run_all_tests.py
    python tests/run_all_tests.py --verbose
"""

import sys
import os
import unittest
import argparse

# Füge Parent-Directory zum Path hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_tests(verbosity=2):
    """
    Führt alle Tests aus.
    
    Args:
        verbosity: Test output verbosity (1=minimal, 2=normal, 3=detailed)
    """
    # Discover und lade alle Tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Führe Tests aus
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Ausgabe Zusammenfassung
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:  {len(result.failures)}")
    print(f"Errors:    {len(result.errors)}")
    print(f"Skipped:   {len(result.skipped)}")
    print("="*70)
    
    # Return Code
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(
        description='Run all tests for Vision-to-Grasp Pipeline'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Increase test output verbosity'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Decrease test output verbosity'
    )
    
    args = parser.parse_args()
    
    # Bestimme Verbosity
    if args.quiet:
        verbosity = 1
    elif args.verbose:
        verbosity = 3
    else:
        verbosity = 2
    
    # Führe Tests aus
    sys.exit(run_tests(verbosity))


if __name__ == '__main__':
    main()

