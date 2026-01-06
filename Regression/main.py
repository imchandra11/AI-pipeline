"""
Main entry point for regression training using Lightning CLI.

This script provides the standard Lightning CLI interface for training,
testing, and prediction.
"""

from Regression.cli import RGSLightningCLI


def cli_main():
    """Main function to run Lightning CLI."""
    # Use RGSLightningCLI instead of LightningCLI for compatibility with .yaml
    cli = RGSLightningCLI()


if __name__ == "__main__":
    cli_main()

