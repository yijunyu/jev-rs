#!/usr/bin/env sh
# jev-rs installer.  Usage:
#   curl -fsSL https://raw.githubusercontent.com/yijunyu/jev-rs/main/install.sh | sh
# Options (env): JEV_VERSION=v0.1.0  JEV_INSTALL_DIR=~/.local/bin  JEV_FROM_SOURCE=1
set -eu

REPO="yijunyu/jev-rs"
INSTALL_DIR="${JEV_INSTALL_DIR:-$HOME/.local/bin}"
VERSION="${JEV_VERSION:-latest}"

say() { printf '%s\n' "$*" >&2; }
die() { say "install.sh: $*"; exit 1; }

os=$(uname -s)
arch=$(uname -m)
case "$os" in
  Darwin) sys=apple-darwin ;;
  Linux)  sys=unknown-linux-gnu ;;
  *) die "unsupported OS: $os (build from source: cargo install --git https://github.com/$REPO)" ;;
esac
case "$arch" in
  arm64|aarch64) cpu=aarch64 ;;
  x86_64|amd64)  cpu=x86_64 ;;
  *) die "unsupported CPU: $arch" ;;
esac
target="$cpu-$sys"

from_source() {
  command -v cargo >/dev/null 2>&1 || die "no prebuilt binary for $target and cargo is not installed; install Rust from https://rustup.rs and rerun"
  say "building from source with cargo (this takes a minute)…"
  cargo install --locked --git "https://github.com/$REPO" --root "$(dirname "$INSTALL_DIR")" jev-rs
}

mkdir -p "$INSTALL_DIR"
if [ "${JEV_FROM_SOURCE:-0}" = "1" ]; then
  from_source
else
  if [ "$VERSION" = "latest" ]; then
    url="https://github.com/$REPO/releases/latest/download/jev-$target.tar.gz"
  else
    url="https://github.com/$REPO/releases/download/$VERSION/jev-$target.tar.gz"
  fi
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  say "downloading $url"
  if curl -fsSL "$url" -o "$tmp/jev.tar.gz"; then
    tar -xzf "$tmp/jev.tar.gz" -C "$tmp"
    install -m 0755 "$tmp/jev-$target/jev" "$INSTALL_DIR/jev"
  else
    say "no prebuilt binary for $target at $url"
    from_source
  fi
fi

say ""
say "installed: $INSTALL_DIR/jev  ($("$INSTALL_DIR/jev" --version 2>/dev/null || echo jev))"
case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *) say "add to PATH:  export PATH=\"$INSTALL_DIR:\$PATH\"" ;;
esac
if ! command -v llama-server >/dev/null 2>&1; then
  say ""
  say "jev needs a llama-server with any GGUF model. Install one with:"
  say "  brew install llama.cpp     # macOS"
  say "  see https://github.com/ggml-org/llama.cpp/releases for Linux binaries"
fi
say ""
say "next:  llama-server -hf Qwen/Qwen3-4B-GGUF --port 8089 -np 2 -c 8192"
say "       jev --backend http://127.0.0.1:8089 ask --state 'Charged twice, refund me today' --noul 'refund=Is a refund requested?'"
