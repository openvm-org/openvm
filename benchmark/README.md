# AFS Benchmark

## Configuration

### `--config-folder` folder setting

Setting a `--config-folder` will get benchmark utility to read all .toml files from that folder and parse each as a `PageConfig`. For each `PageConfig` parsed, it will run the benchmark with the configuration and output it to a csv file in `benchmark/output`.

### `--percent-writes`

Percentage (where 100 = 100%) of config file's `max_rw_ops` that are writes to the database. Will create random `INSERT` commands up to the page height, and then create `WRITE` instructions for the remaining values.

Note that `--percent-reads` and `--percent-writes` must be less than or equal to 100, but do not need to total 100.

### `--percent-reads`

Percentage (where 100 = 100%) of config file's `max_rw_ops` that are `READ`s. Note that there must be at least one value already inserted.

## Commands

Run these commands from the root of the repository

```bash
cargo run --bin benchmark -- rw --config-folder benchmark/configs/rw -r 50 -w 50
```
