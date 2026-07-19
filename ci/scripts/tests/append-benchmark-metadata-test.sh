#!/usr/bin/env bash

set -euo pipefail

readonly script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
temp_dir="$(mktemp -d "${TMPDIR:-/tmp}/append-benchmark-metadata-test.XXXXXX")"
trap 'rm -rf "$temp_dir"' EXIT

markdown="$temp_dir/sha2-bench.md"
cat > "$markdown" <<'EOF'
# sha2-bench

![GPU Memory Usage](sha2-bench.memory.svg)
EOF
mkdir -p "$temp_dir/flamegraphs"
printf '<svg>first</svg>\n' > "$temp_dir/flamegraphs/sha2-bench-cpu.svg"
printf '<svg>second</svg>\n' > "$temp_dir/flamegraphs/sha2-bench-gpu.svg"

bash "$script_dir/append-benchmark-metadata.sh" \
    --markdown "$markdown" \
    --instance-type "g7e.2xlarge" \
    --memory-allocator "jemalloc" \
    --commit-url "https://example.com/commit/abc" \
    --workflow-url "https://example.com/actions/runs/123" \
    --max-segment-length "4194304" \
    --memory-svg-url "https://example.com/charts/sha2-bench.memory.svg" \
    --flamegraph-dir "$temp_dir/flamegraphs" \
    --flamegraph-link-prefix "flamegraphs/sha2-bench" \
    --peak-gpu-memory-gib "12.50"

expected="$temp_dir/expected.md"
cat > "$expected" <<'EOF'
# sha2-bench

![GPU Memory Usage](https://example.com/charts/sha2-bench.memory.svg)

<details>
<summary>Flamegraphs</summary>

[![](flamegraphs/sha2-bench/sha2-bench-cpu.svg)](flamegraphs/sha2-bench/sha2-bench-cpu.svg)
[![](flamegraphs/sha2-bench/sha2-bench-gpu.svg)](flamegraphs/sha2-bench/sha2-bench-gpu.svg)

</details>

Commit: https://example.com/commit/abc

Max Segment Length: 4194304

Instance Type: g7e.2xlarge

Memory Allocator: jemalloc

[Benchmark Workflow](https://example.com/actions/runs/123)

**Peak GPU Memory (nvidia-smi):** 12.50 GB
EOF

diff -u "$expected" "$markdown"

without_network="$temp_dir/without-network.md"
printf '# no network\n' > "$without_network"
PATH=/usr/bin:/bin bash "$script_dir/append-benchmark-metadata.sh" \
    --markdown "$without_network" \
    --instance-type "c8i.2xlarge" \
    --memory-allocator "mimalloc" \
    --commit-url "https://example.com/commit/def" \
    --workflow-url "https://example.com/actions/runs/456"
grep -Fq 'Instance Type: c8i.2xlarge' "$without_network"

if bash "$script_dir/append-benchmark-metadata.sh" \
    --markdown "$without_network" \
    --instance-type "c8i.2xlarge" \
    --memory-allocator "mimalloc" \
    --commit-url "https://example.com/commit/def" \
    --workflow-url "https://example.com/actions/runs/456" \
    --flamegraph-dir "$temp_dir/flamegraphs"; then
    echo "Expected a missing flamegraph link prefix to fail" >&2
    exit 1
fi

echo "append-benchmark-metadata tests passed"
