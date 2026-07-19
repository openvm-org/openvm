#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: append-benchmark-metadata.sh \
  --markdown PATH \
  --instance-type TYPE \
  --memory-allocator ALLOCATOR \
  --commit-url URL \
  --workflow-url URL \
  [--max-segment-length VALUE] \
  [--memory-svg-url URL] \
  [--flamegraph-dir PATH] \
  [--flamegraph-link-prefix URL_OR_PATH] \
  [--peak-gpu-memory-gib VALUE]
EOF
}

markdown=""
instance_type=""
memory_allocator=""
commit_url=""
workflow_url=""
max_segment_length=""
memory_svg_url=""
flamegraph_dir=""
flamegraph_link_prefix=""
peak_gpu_memory_gib=""
temp_markdown=""

cleanup() {
    if [[ -n "$temp_markdown" ]]; then
        rm -f "$temp_markdown"
    fi
}
trap cleanup EXIT

while (($#)); do
    case "$1" in
        --markdown) markdown="${2:?missing value for $1}"; shift 2 ;;
        --instance-type) instance_type="${2:?missing value for $1}"; shift 2 ;;
        --memory-allocator) memory_allocator="${2:?missing value for $1}"; shift 2 ;;
        --commit-url) commit_url="${2:?missing value for $1}"; shift 2 ;;
        --workflow-url) workflow_url="${2:?missing value for $1}"; shift 2 ;;
        --max-segment-length) max_segment_length="${2:?missing value for $1}"; shift 2 ;;
        --memory-svg-url) memory_svg_url="${2:?missing value for $1}"; shift 2 ;;
        --flamegraph-dir) flamegraph_dir="${2:?missing value for $1}"; shift 2 ;;
        --flamegraph-link-prefix) flamegraph_link_prefix="${2:?missing value for $1}"; shift 2 ;;
        --peak-gpu-memory-gib) peak_gpu_memory_gib="${2:?missing value for $1}"; shift 2 ;;
        -h | --help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

for required in markdown instance_type memory_allocator commit_url workflow_url; do
    if [[ -z "${!required}" ]]; then
        echo "Missing required argument: --${required//_/-}" >&2
        usage
        exit 2
    fi
done
if [[ ! -f "$markdown" ]]; then
    echo "Markdown file does not exist: $markdown" >&2
    exit 1
fi
if [[ -n "$flamegraph_dir" && -z "$flamegraph_link_prefix" ]]; then
    echo "--flamegraph-link-prefix is required with --flamegraph-dir" >&2
    exit 2
fi

if [[ -n "$memory_svg_url" ]]; then
    svg_path="${markdown%.md}.memory.svg"
    svg_reference="![GPU Memory Usage]($(basename "$svg_path"))"
    replacement="![GPU Memory Usage](${memory_svg_url})"
    temp_markdown="$(mktemp "${markdown}.XXXXXX")"
    while IFS= read -r line || [[ -n "$line" ]]; do
        if [[ "$line" == "$svg_reference" ]]; then
            printf '%s\n' "$replacement"
        else
            printf '%s\n' "$line"
        fi
    done < "$markdown" > "$temp_markdown"
    cat "$temp_markdown" > "$markdown"
    rm -f "$temp_markdown"
    temp_markdown=""
fi

if [[ -n "$flamegraph_dir" && -d "$flamegraph_dir" ]]; then
    shopt -s nullglob
    flamegraphs=("$flamegraph_dir"/*.svg)
    shopt -u nullglob
    if ((${#flamegraphs[@]})); then
        {
            printf '\n<details>\n<summary>Flamegraphs</summary>\n\n'
            for flamegraph in "${flamegraphs[@]}"; do
                filename="$(basename "$flamegraph")"
                link="${flamegraph_link_prefix%/}/${filename}"
                printf '[![](%s)](%s)\n' "$link" "$link"
            done
            printf '\n</details>\n'
        } >> "$markdown"
    fi
fi

{
    printf '\nCommit: %s\n' "$commit_url"
    if [[ -n "$max_segment_length" ]]; then
        printf '\nMax Segment Length: %s\n' "$max_segment_length"
    fi
    printf '\nInstance Type: %s\n' "$instance_type"
    printf '\nMemory Allocator: %s\n' "$memory_allocator"
    printf '\n[Benchmark Workflow](%s)\n' "$workflow_url"
    if [[ -n "$peak_gpu_memory_gib" && "$peak_gpu_memory_gib" != "0" ]]; then
        printf '\n**Peak GPU Memory (nvidia-smi):** %s GB\n' "$peak_gpu_memory_gib"
    fi
} >> "$markdown"
