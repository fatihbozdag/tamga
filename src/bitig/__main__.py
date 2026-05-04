"""Allow `python -m bitig ...` to invoke the CLI."""

from bitig.cli import app

if __name__ == "__main__":
    app()
