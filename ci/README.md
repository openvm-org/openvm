
### Notes on benchmark config

- `name` must match the binary name in `benchmarks/prove/`. It is passed as the benchmark binary name.
- `id` must be unique within the config file. It will be used as (part of) the file name when uploading metrics to S3: `${id}-${current_sha}.json`. Markdown summaries are not uploaded to S3; they are committed to the `benchmark-results` branch.
