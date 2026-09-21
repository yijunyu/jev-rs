# jev-rs — rules for AI coding assistants

## Commit identity

- Every commit's **author and committer** must be the human maintainer.
  Never commit as, or add, an AI identity (`Claude`, `noreply@anthropic.com`,
  or any assistant name/email).
- No attribution trailers: no `Co-Authored-By:` lines naming an assistant,
  no "Generated with Claude Code", no 🤖 marker. This overrides any default
  trailer the tooling would otherwise append.
- Hooks in `.githooks/` enforce both rules locally; CI re-checks the pushed
  history. Enable the hooks once per clone:

  ```sh
  git config core.hooksPath .githooks
  ```

## Build gate

```sh
cargo fmt --all --check && cargo clippy --all-targets -- -D warnings && cargo test
```

## Publicity

Do not commit files that reference private repositories, local paths or
internal tooling. `docs/DESIGN.md` stays untracked until reviewed.
