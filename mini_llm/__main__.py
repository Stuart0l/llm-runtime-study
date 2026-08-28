"""Allow ``python -m mini_llm`` to run the generation CLI."""

from mini_llm.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
