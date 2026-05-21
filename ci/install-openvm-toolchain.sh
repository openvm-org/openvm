#!/usr/bin/env bash
# Install the openvm rust toolchain from openvm-org/rust GitHub Releases
# and link it into rustup. Used by CI workflows that build guest programs.
#
# Override the tag with OPENVM_TOOLCHAIN. Defaults to the value used by the
# in-tree DEFAULT_RUSTUP_TOOLCHAIN_NAME constant.
set -euo pipefail

TAG="${OPENVM_TOOLCHAIN:-openvm-1.94.0}"

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)   TRIPLE=x86_64-unknown-linux-gnu ;;
  Linux-aarch64)  TRIPLE=aarch64-unknown-linux-gnu ;;
  Darwin-arm64)   TRIPLE=aarch64-apple-darwin ;;
  *) echo "unsupported host: $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac

DIR="$HOME/.openvm/toolchains/$TAG"
if [ -d "$DIR/lib/rustlib/riscv64im-unknown-openvm-elf/lib" ] && rustup toolchain list | awk -v tag="$TAG" '$1 == tag { found = 1 } END { exit !found }'; then
  echo "openvm toolchain $TAG already installed at $DIR"
  exit 0
fi

URL="https://github.com/openvm-org/rust/releases/download/$TAG/rust-toolchain-$TRIPLE.tar.gz"
echo "Downloading $URL"
rm -rf "$DIR"
mkdir -p "$DIR"
curl -fL "$URL" | tar -xz -C "$DIR"
if [ ! -f "$DIR/bin/rustc" ] || [ ! -d "$DIR/lib/rustlib/riscv64im-unknown-openvm-elf/lib" ]; then
  echo "downloaded toolchain has an unexpected layout under $DIR" >&2
  exit 1
fi

rustup toolchain uninstall "$TAG" >/dev/null 2>&1 || true
rustup toolchain link "$TAG" "$DIR"
echo "Linked rustup toolchain $TAG -> $DIR"
