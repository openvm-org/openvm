names=$(echo '[{"name":"verify_fibair","app_log_blowup":1,"agg_log_blowup":1},{"name":"verify_fibair","app_log_blowup":2,"agg_log_blowup":2},{"name":"fibonacci","app_log_blowup":2,"agg_log_blowup":2},{"name":"revm_transfer","app_log_blowup":2,"agg_log_blowup":2},{"name":"regex","app_log_blowup":2,"agg_log_blowup":2},{"name":"base64_json","app_log_blowup":2,"agg_log_blowup":2}]' | jq -r '.[].name' | sort -u)
echo $names
while read name; do
    echo "Processing results for benchmark: $name"
done <<< "$names"