
### Notes on benchmark config

- `name` must match the binary name in `benchmarks/prove/`. It is passed as the benchmark binary name.
- `id` must be unique within the config file. It will be used as (part of) the file name when uploading to S3: `${id}-${current_sha}.[md/json]`
