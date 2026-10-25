"""Compatibility entry point for the validated experiment driver."""
try:
    from .run_experiments import main
except ImportError:
    from run_experiments import main

if __name__ == "__main__":
    raise SystemExit(main())
