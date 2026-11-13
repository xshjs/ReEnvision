"""Compatibility entrypoint: Repair-only training; original inference unchanged."""
from legacy_training import main

if __name__ == "__main__":
    main()
