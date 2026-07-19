#!/usr/bin/env bash

set -euo pipefail

readonly toolchain_date="${OPENVM_RUST_TOOLCHAIN_DATE:-2026-01-18}"

case "$(uname -m)" in
    arm64 | aarch64)
        target="aarch64-unknown-linux-gnu"
        ;;
    x86_64 | amd64)
        target="x86_64-unknown-linux-gnu"
        ;;
    *)
        echo "Unsupported architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

rustup component add rust-src --toolchain "nightly-${toolchain_date}-${target}"
