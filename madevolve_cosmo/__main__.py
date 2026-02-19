"""CLI entry point for MadEvolve-Cosmo.

This is a thin wrapper around the MadEvolve CLI with cosmology-specific branding.
All evolution functionality is provided by the madevolve package.
"""

from __future__ import annotations

import sys


def print_cosmo_banner():
    """Print MadEvolve-Cosmo banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                    MadEvolve-Cosmo                        ║
    ║     LLM-Driven Evolution for Cosmology Applications       ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def main() -> int:
    """Main entry point for MadEvolve-Cosmo CLI."""
    from madevolve.__main__ import main as madevolve_main

    # Print cosmology-specific banner before delegating to MadEvolve
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        if "--quiet" not in sys.argv:
            print_cosmo_banner()
            # Add --quiet to prevent MadEvolve from printing its own banner
            sys.argv.append("--quiet")

    return madevolve_main()


if __name__ == "__main__":
    sys.exit(main())
