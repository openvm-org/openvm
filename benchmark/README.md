# AFS Benchmark

## Configuration

### `--config-folder` folder setting

Setting a `--config-folder` will get benchmark utility to read all .toml files from that folder and parse each as a `PageConfig`. For each `PageConfig` parsed, it will run the benchmark with the configuration and output it to a csv file in `benchmark/output`.

### `--percent-writes`

Percentage (where 100 = 100%) of config file's `max_rw_ops` that are writes to the database. Will create random `INSERT` commands up to the page height, and then create `WRITE` instructions for the remaining values.

Note that `--percent-reads` and `--percent-writes` must be less than or equal to 100, but do not need to total 100.

### `--percent-reads`

Percentage (where 100 = 100%) of config file's `max_rw_ops` that are `READ`s. Note that there must be at least one value already inserted.

## `PageConfig` generation

Run the test `run_generate_configs()` located in `benchmark/src/utils/config_gen.rs`, which will generate configs in `benchmark/config/rw`. You can optionally pass in these generated configs with the flag `--config-folder benchmark/config/rw`. If no folder is passed in, the config permutations will be generated on the fly in run from memory.

## Benchmarks

We generate the following benchmark data for each run:

- preprocessed: Total width of preprocessed AIR
- main: Total width of partitioned main AIR
- challenge: Total width of after challenge AIR
- keygen_time: Keygen time: Time to generate keys
- cache_time: Cache time: Time to generate cached trace
- prove_time: Total time to generate the proof prove (inclusive of all prove timing items above)
  - prove_load_trace_gen: Time to generate load_page_and_ops trace
  - prove_load_trace_commit: Time to commit load_page_and_ops trace
  - prove_generate: Time to generate trace
  - prove_commit: Time to commit trace
- verify_time: Total time to verify the proof

## Commands

Run these commands from the root of the repository

```bash
cargo run --release --bin benchmark -- rw -r 90 -w 10
```
