# Contributing

Keep each branch focused on one change. Before opening a pull request, explain the
problem, the chosen behavior, and any user-facing compatibility effect.

## Checks

Install the development dependencies and run the tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

Run Ruff on every Python file you change:

```bash
ruff check path/to/changed_file.py
ruff format --check path/to/changed_file.py
```

Add or update tests when behavior changes. Documentation-only changes should still
check referenced paths and commands.

## Repository hygiene

Do not commit API keys, passwords, access tokens, private data, local datasets,
model checkpoints, or generated caches. Discuss large files with the maintainers
before adding them; prefer a stable external source when redistribution is not
allowed.

Use `Co-authored-by: Name <email>` trailers only for people who contributed to the
committed work. Spell names and email addresses as each co-author requests, and do
not add a trailer for review alone.
