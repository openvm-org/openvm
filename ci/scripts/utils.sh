
get_metric_name() {
    local is_e2e="$1"
    local bin_name="$2"
    local app_log_blowup="$3"
    local agg_log_blowup="$4"
    local root_log_blowup="$5"
    local internal_log_blowup="$6"
    local max_segment_length="$7"
    local instance_type="$8"
    local memory_allocator="$9"

    local metric_name="${bin_name}-${app_log_blowup}-${agg_log_blowup}"
    
    if [[ "$is_e2e" == "true" ]]; then
        metric_name="${metric_name}-${root_log_blowup}-${internal_log_blowup}"
    fi
    
    metric_name="${metric_name}-${max_segment_length}-${instance_type}-${memory_allocator}"
    
    echo "$metric_name"
}

generate_markdown() {
    local metric_path="$1"
    local metric_name="$2"
    local current_sha="$3"
    local s3_metrics_path="$4"

    if [[ -f $metric_path ]]; then
        s5cmd cp $metric_path $s3_metrics_path/${current_sha}-${metric_name}.json

        prev_path="${s3_metrics_path}/main-${metric_name}.json"
        count=`s5cmd ls $prev_path | wc -l`

        if [[ $count -gt 0 ]]; then
            s5cmd cp $prev_path prev.json
            python3 ci/scripts/metric_unify/main.py $metric_path --prev prev.json --aggregation-json ci/scripts/metric_unify/aggregation.json > results.md
        else
            echo "No previous benchmark on main branch found"
            python3 ci/scripts/metric_unify/main.py $metric_path --aggregation-json ci/scripts/metric_unify/aggregation.json > results.md
        fi
    else
        echo "No benchmark metrics found at ${metric_path}"
    fi
}

add_metadata() {
    local result_path="$2"
    local max_segment_length="$5"
    local instance_type="$6"
    local memory_allocator="$7"
    local repo="$8"
    local run_id="$9"

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
    s5cmd cp $result_path "${S3_PATH}/${current_sha}-${METRIC_NAME}.md"
}


