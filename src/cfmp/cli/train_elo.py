"""Backward-compatible entry point. The rating is the margin model, not win-loss Elo."""

from cfmp.cli.train import main

if __name__ == "__main__":
    main()
