#!/usr/bin/env bash

set -euo pipefail

readonly script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
temp_dir="$(mktemp -d "${TMPDIR:-/tmp}/install-s5cmd-test.XXXXXX")"
trap 'rm -rf "$temp_dir"' EXIT

mkdir -p "$temp_dir/matching"
cat > "$temp_dir/matching/s5cmd" <<'EOF'
#!/usr/bin/env bash
echo 'v2.2.2-existing'
EOF
chmod +x "$temp_dir/matching/s5cmd"
: > "$temp_dir/matching-github-path"
PATH="$temp_dir/matching:/usr/bin:/bin" \
    GITHUB_PATH="$temp_dir/matching-github-path" \
    bash "$script_dir/install-s5cmd.sh" > "$temp_dir/matching-output"
grep -Fq 's5cmd 2.2.2 is already installed' "$temp_dir/matching-output"
[[ ! -s "$temp_dir/matching-github-path" ]]

mkdir -p "$temp_dir/mocks" "$temp_dir/install" "$temp_dir/runner-temp"
cat > "$temp_dir/mocks/s5cmd" <<'EOF'
#!/usr/bin/env bash
echo 'v1.0.0-stale'
EOF
cat > "$temp_dir/mocks/uname" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in
    -s) echo Linux ;;
    -m) echo x86_64 ;;
    *) exit 2 ;;
esac
EOF
cat > "$temp_dir/mocks/curl" <<'EOF'
#!/usr/bin/env bash
output=""
while (($#)); do
    case "$1" in
        --output) output="$2"; shift 2 ;;
        *) shift ;;
    esac
done
case "$output" in
    */s5cmd_checksums.txt)
        printf '%s\n' \
            '0000  s5cmd_2.2.2_Linux-arm64.tar.gz' \
            '1111  s5cmd_2.2.2_Linux-64bit.tar.gz' > "$output"
        ;;
    *) printf 'archive fixture\n' > "$output" ;;
esac
EOF
cat > "$temp_dir/mocks/sha256sum" <<'EOF'
#!/usr/bin/env bash
cat > "$CHECKSUM_STDIN"
EOF
cat > "$temp_dir/mocks/tar" <<'EOF'
#!/usr/bin/env bash
destination=""
while (($#)); do
    case "$1" in
        -C) destination="$2"; shift 2 ;;
        *) shift ;;
    esac
done
cat > "$destination/s5cmd" <<'INNER'
#!/usr/bin/env bash
echo 'v2.2.2-installed'
INNER
chmod +x "$destination/s5cmd"
EOF
chmod +x "$temp_dir/mocks/"*
: > "$temp_dir/mismatch-github-path"
PATH="$temp_dir/mocks:/usr/bin:/bin" \
    CHECKSUM_STDIN="$temp_dir/checksum-stdin" \
    GITHUB_PATH="$temp_dir/mismatch-github-path" \
    RUNNER_TEMP="$temp_dir/runner-temp" \
    S5CMD_INSTALL_DIR="$temp_dir/install" \
    bash "$script_dir/install-s5cmd.sh" > "$temp_dir/mismatch-output"

grep -Fq 'Ignoring unpinned s5cmd' "$temp_dir/mismatch-output"
grep -Fxq '1111  s5cmd_2.2.2_Linux-64bit.tar.gz' "$temp_dir/checksum-stdin"
grep -Fxq "$temp_dir/install" "$temp_dir/mismatch-github-path"
[[ "$($temp_dir/install/s5cmd version)" == 'v2.2.2-installed' ]]

echo "install-s5cmd tests passed"
