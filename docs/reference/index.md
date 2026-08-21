# API Reference

These pages are generated directly from the NumPy-style docstrings in
`src/rag/**` via [mkdocstrings](https://mkdocstrings.github.io/), grouped
to match the package layout described in `CLAUDE.md`'s directory map.
They document the public interfaces (base classes, factory functions,
pipeline entry points); private (`_leading_underscore`) helpers are
omitted.

Nothing on these pages is hand-maintained text: if a docstring is wrong or
missing, fix it at the source (the `.py` file) and rebuild, rather than
editing here.

!!! note
    Config field docs describe *shape*, not live values: none of these
    pages embed actual `.env` contents, database hosts, or API keys. Every
    secret-shaped field in the codebase is named `<field>_env_var` and
    resolved from the process environment at runtime, never hardcoded.
