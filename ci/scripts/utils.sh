add_metadata_and_flamegraphs() {
    local metric_path="$1"
    local md_path="$2"
    local matrix="$3"
    local commit_url="$4"
    local benchmark_workflow_url="$5"
    # vars: $FLAMEGRAPHS, $S3_PUBLIC_PATH_BASE, $S3_PUBLIC_URL_BASE, $CURRENT_SHA

    id=${metric_path%%-*} # first part before -
    echo "id: $id"

    inputs=$(echo "$matrix" | jq -r --arg id "$id" '.[] |
      select(.id == $id) |
      {
        max_segment_length: .max_segment_length,
        instance_type: .instance_type,
        memory_allocator: .memory_allocator
      }')
    echo "inputs: $inputs"

    # Upload memory chart SVG to S3 and update md link
    local svg_path="${metric_path%.json}.memory.svg"
    if [ -f "$svg_path" ]; then
      benchmark_name=$(basename "$metric_path" | cut -d'-' -f1)
      s3_svg_path="${S3_PUBLIC_PATH_BASE}/charts/${benchmark_name}-${CURRENT_SHA}/$(basename "$svg_path")"
      s5cmd cp "$svg_path" "$s3_svg_path"
      svg_url="${S3_PUBLIC_URL_BASE}/charts/${benchmark_name}-${CURRENT_SHA}/$(basename "$svg_path")"
      # Replace local SVG reference with S3 URL in the markdown
      sed -i "s|!\[GPU Memory Usage\]($(basename "$svg_path"))|![GPU Memory Usage]($svg_url)|g" "$md_path"
    fi

    if [ ! -z "$inputs" ]; then
      max_segment_length=$(echo "$inputs" | jq -r '.max_segment_length')
      instance_type=$(echo "$inputs" | jq -r '.instance_type')
      memory_allocator=$(echo "$inputs" | jq -r '.memory_allocator')

      # Call add_metadata for each file with its corresponding data
      add_metadata \
        "$md_path" \
        "$max_segment_length" \
        "$instance_type" \
        "$memory_allocator" \
        "$commit_url" \
        "$benchmark_workflow_url"
    fi
}

add_metadata() {
    local result_path="$1"
    local max_segment_length="$2"
    local instance_type="$3"
    local memory_allocator="$4"
    local commit_url="$5"
    local benchmark_workflow_url="$6"
    # vars: $FLAMEGRAPHS, $S3_PUBLIC_PATH_BASE, $S3_PUBLIC_URL_BASE, $CURRENT_SHA

    echo "" >> $result_path
    if [[ "$FLAMEGRAPHS" == 'true' ]]; then
        echo "<details>" >> $result_path
        echo "<summary>Flamegraphs</summary>" >> $result_path
        echo "" >> $result_path
        benchmark_name=$(basename "$result_path" | cut -d'-' -f1)
        flamegraph_files=$(s5cmd ls ${S3_PUBLIC_PATH_BASE}/flamegraphs/${benchmark_name}-${CURRENT_SHA}/*.svg | awk '{print $4}' | xargs -n1 basename)
        for file in $flamegraph_files; do
            flamegraph_url=${S3_PUBLIC_URL_BASE}/flamegraphs/${benchmark_name}-${CURRENT_SHA}/${file}
            echo "[![]($flamegraph_url)]($flamegraph_url)" >> $result_path
        done
        echo "" >> $result_path
        echo "</details>" >> $result_path
        echo "" >> $result_path
    fi
    echo "Commit: ${commit_url}" >> $result_path
    echo "" >> $result_path
    echo "Max Segment Length: $max_segment_length" >> $result_path
    echo "" >> $result_path
    echo "Instance Type: $instance_type" >> $result_path
    echo "" >> $result_path
    echo "Memory Allocator: $memory_allocator" >> $result_path
    echo "" >> $result_path
    echo "[Benchmark Workflow](${benchmark_workflow_url})" >> $result_path
}

commit_and_push_benchmark_results() {
    local files=$1
    local commit_message=$2
    git add ${files}
    git commit --allow-empty -m "${commit_message}"

    MAX_RETRIES=10
    RETRY_DELAY=5
    ATTEMPT=0
    SUCCESS=false

    while [ $ATTEMPT -lt $MAX_RETRIES ]; do
        echo "Attempt $((ATTEMPT + 1)) to push of $MAX_RETRIES..."
        git fetch origin benchmark-results
        git merge origin/benchmark-results --no-edit
        if git push origin benchmark-results; then
            SUCCESS=true
            break
        else
            echo "Push failed. Retrying in $RETRY_DELAY seconds..."
            sleep $RETRY_DELAY
            ATTEMPT=$((ATTEMPT + 1))
        fi
    done

    if [ "$SUCCESS" = false ]; then
        echo "PUSH_FAILED"
        exit 1
    fi
}

install_s5cmd() {
    arch=$(uname -m)
    case $arch in
    arm64|aarch64)
        rustup component add rust-src --toolchain nightly-2025-08-02-aarch64-unknown-linux-gnu
        S5CMD_BIN="s5cmd_2.2.2_linux_arm64.deb"
        ;;
    x86_64|amd64)
        rustup component add rust-src --toolchain nightly-2025-08-02-x86_64-unknown-linux-gnu
        S5CMD_BIN="s5cmd_2.2.2_linux_amd64.deb"
        ;;
    *)
        echo "Unsupported architecture: $arch"
        exit 1
        ;;
    esac

    echo "Checking s5cmd"
    if type s5cmd &>/dev/null; then
        echo "s5cmd was installed."
    else
        TMP_DIR=/tmp/s5cmd
        rm -rf $TMP_DIR
        mkdir $TMP_DIR
        echo "s5cmd was not installed. Installing.."
        wget "https://github.com/peak/s5cmd/releases/download/v2.2.2/${S5CMD_BIN}" -P $TMP_DIR
        sudo dpkg -i "${TMP_DIR}/${S5CMD_BIN}"
    fi
}
