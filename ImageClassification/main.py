"""
Main entry point for Lightning CLI commands.

This script provides the standard Lightning CLI interface for image classification tasks.
"""

from ImageClassification.cli import IMGLightningCLI


def cli_main():
    """Main function for CLI."""
    CLI = IMGLightningCLI


if __name__ == "__main__":
    cli_main()

