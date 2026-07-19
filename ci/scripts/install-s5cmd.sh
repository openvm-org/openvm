#!/usr/bin/env bash

set -euo pipefail

readonly version="${S5CMD_VERSION:-2.2.2}"
if command -v s5cmd >/dev/null 2>&1; then
    installed_version="$(s5cmd version 2>/dev/null || true)"
    case "$installed_version" in
        "v${version}" | "v${version}-"*)
            echo "s5cmd ${version} is already installed: $(command -v s5cmd)"
            exit 0
            ;;
        *)
            echo "Ignoring unpinned s5cmd at $(command -v s5cmd): ${installed_version:-unknown version}"
            ;;
    esac
fi
case "$(uname -s):$(uname -m)" in
    Linux:x86_64 | Linux:amd64)
        archive="s5cmd_${version}_Linux-64bit.tar.gz"
        ;;
    Linux:arm64 | Linux:aarch64)
        archive="s5cmd_${version}_Linux-arm64.tar.gz"
        ;;
    *)
        echo "Unsupported platform: $(uname -s) $(uname -m)" >&2
        exit 1
        ;;
esac

readonly release_url="https://github.com/peak/s5cmd/releases/download/v${version}"
readonly temp_root="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
readonly install_dir="${S5CMD_INSTALL_DIR:-${temp_root}/openvm-bin}"
temp_dir="$(mktemp -d "${temp_root%/}/s5cmd.XXXXXX")"
trap 'rm -rf "$temp_dir"' EXIT

curl --fail --location --silent --show-error \
    --output "${temp_dir}/${archive}" \
    "${release_url}/${archive}"
curl --fail --location --silent --show-error \
    --output "${temp_dir}/s5cmd_checksums.txt" \
    "${release_url}/s5cmd_checksums.txt"

(
    cd "$temp_dir"
    grep -E "[[:space:]]${archive}$" s5cmd_checksums.txt | sha256sum --check --strict -
)

tar -xzf "${temp_dir}/${archive}" -C "$temp_dir" s5cmd
mkdir -p "$install_dir"
install -m 0755 "${temp_dir}/s5cmd" "${install_dir}/s5cmd"

if [[ -n "${GITHUB_PATH:-}" ]]; then
    printf '%s\n' "$install_dir" >> "$GITHUB_PATH"
else
    echo "Installed s5cmd to ${install_dir}; add this directory to PATH."
fi
