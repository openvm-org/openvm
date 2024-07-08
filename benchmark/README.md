# AFS Benchmark

## Configuration

### `--percent-reads`

Percentage (where 100 = 100%) of config file's `max_rw_ops` that are `READ`s.

### `--percent-writes`

Percentage (where 100 = 100%) of config file's `max_rw_ops` that are writes to the database. Will create `INSERT` values up to the page height, and then create `WRITE` instructions for the remaining values.

Note that `--percent-reads` and `--percent-writes` must be less than or equal to 100, but do not need to total 100.

## Commands

Run these commands from the root of the repository

```bash
cargo run --bin benchmark -- rw --config-folder benchmark/configs/config.toml --output-file benchmark/output/output.csv -r 50 -w 50
```
