"""Allow ``python -m fpc`` to invoke the GenoCortex CLI."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
