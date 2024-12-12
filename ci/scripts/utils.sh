generate_markdown() {
    local metric_path="$1"
    local metric_name="$2"
    local s3_metrics_path="$3"
    local afs_root="$4"

    if [[ -f $metric_path ]]; then
        prev_path="${s3_metrics_path}/main-${metric_name}.json"
        count=`s5cmd ls $prev_path | wc -l`

        if [[ $count -gt 0 ]]; then
            s5cmd cp $prev_path prev.json
            python3 ${afs_root}/ci/scripts/metric_unify/main.py $metric_path --prev prev.json --aggregation-json ${afs_root}/ci/scripts/metric_unify/aggregation.json > results.md
        else
            echo "No previous benchmark on main branch found"
            python3 ${afs_root}/ci/scripts/metric_unify/main.py $metric_path --aggregation-json ${afs_root}/ci/scripts/metric_unify/aggregation.json > results.md
        fi
    else
        echo "No benchmark metrics found at ${metric_path}"
    fi
}

add_metadata() {
    local result_path="$1"
    local max_segment_length="$2"
    local instance_type="$3"
    local memory_allocator="$4"
    local repo="$5"
    local run_id="$6"

    commit_url="https://github.com/${repo}/commit/${current_sha}"
    echo "" >> $result_path
    if [[ "$UPLOAD_FLAMEGRAPHS" == '1' ]]; then
        echo "<details>" >> $result_path
        echo "<summary>Flamegraphs</summary>" >> $result_path
        echo "" >> $result_path
        for file in .bench_metrics/flamegraphs/*.svg; do
        filename=$(basename "$file")
            flamegraph_url=https://axiom-public-data-sandbox-us-east-1.s3.us-east-1.amazonaws.com/benchmark/github/flamegraphs/${current_sha}/${filename}
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
    echo "[Benchmark Workflow](https://github.com/${repo}/actions/runs/${run_id})" >> $result_path
}

commit_and_push_gh_pages() {
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
        git fetch origin gh-pages
        git merge origin/gh-pages --no-edit
        if git push origin gh-pages; then
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
        rustup component add rust-src --toolchain nightly-2024-10-30-aarch64-unknown-linux-gnu
        S5CMD_BIN="s5cmd_2.2.2_linux_arm64.deb"
        ;;
    x86_64|amd64)
        rustup component add rust-src --toolchain nightly-2024-10-30-x86_64-unknown-linux-gnu
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