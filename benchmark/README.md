# AFS Benchmark

## Configuration

### Benchmark data

Each benchmark has a `BenchmarkData` struct that is defined in `src/config/benchmark_data.rs`. For example, the ReadWrite benchmark data is defined in `benchmark_data_rw()` in that file. This is passed into the `benchmark_execute` call in `src/cli/mod.rs`. If you wish to set up different filters for events or timing, you can add your tracing via the `info!`, or `info_span!` macros from the `tracing` crate.

You'll need to add any desired events/timings to the event filters (for `info!`) or timing filters (for `info_span!`). You can add the appropriate header and the log filter string to isolate the timing value. For the log filter string, please be sure it is specific enough to capture only the desired timing value.

### Runtime `PageConfig` generation

Configurations are generated at runtime from the `generate_configs` function in `src/config/config_gen.rs`. Some items are commented out in order to speed up benchmarking since the generator creates all permutations of each vec. Uncomment items as necessary for your benchmark use case.

### `--config-folder` folder setting

Setting a `--config-folder` will skip `generate_configs` and will instead read all .toml files from that folder and parse each as a `PageConfig`. For each `PageConfig` parsed, it will run the benchmark with the configuration and output it to a csv file in `benchmark/output`.

## ReadWrite

To run a single fast benchmark, with faster compile time, run from the root of the repository

```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo run --release --bin benchmark --features parallel -- rw -r 90 -w 10 --config-folder benchmark/config/single_rw
```

To generate a suite of configs and run benchmarks on all of them, run from the root of the repository

```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo run --release --bin benchmark --features parallel -- rw -r 90 -w 10
```

### `--percent-writes` (`-w`)

Percentage (where 100 = 100%) of config file's `max_rw_ops` that are writes to the database. Will create random `INSERT` commands up to the page height, and then create `WRITE` instructions for the remaining values.

Note that `--percent-reads` and `--percent-writes` must be less than or equal to 100, but do not need to total 100.

### `--percent-reads` (`-r`)

Percentage (where 100 = 100%) of config file's `max_rw_ops` that are `READ`s. Note that there must be at least one value already inserted.

## Predicate

Run these commands from the root of the repository

```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo run --release --bin benchmark -- predicate -f benchmark/config/olap/filter_0xfade.afo
```

### `--afo-file` (`-f`)

Pass in an .afo file that contains the predicate instruction. Example .afo file:

```bash
FILTER 0xfade INDEX <= 0xdac0
```

## Predicate

Run this command for multitier read/write

```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo run --release --bin benchmark -- mtrw -n -i 0
```

This command goes through many configs, including large and small page heights. We run benchmarks on the case that n leaves are revealed via n insert operations, where n is 1 or 8. To account for the worst case scenario, we double the leaves in the final tree for certain configs (so 2n). The -n flag tells it to use a new, empty Btree (this should always be on, I haven't implemented large tree yet). The -i flag decides where to start the benchmarking, in terms of config #. To directly jump to worst case testing, setting -i to 8 suffices.

## VM

Run this command for the N-th fibonacci number

```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo run --release --bin benchmark --features parallel -- COMMAND -n N
```

where `COMMAND` is one of the following:

- `vm_fib_program`
- `vm_verify_fibair`
- `vm_fib_verifier_program`

The first command is for the fibonacci program for the N-th fibonacci number. The second command is for the verifier program of FibonacciAir (the hand written air). The last command is for the verifier program of the VM STARK that runs the fibonacci program for N-th fib number.

## Additional test commands

To run only small configs for testing

```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo run --release --bin benchmark -- rw -r 90 -w 10 --config-folder benchmark/config/mini
```

For running tests with only the large configs

```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo run --release --bin benchmark -- rw -r 90 -w 10 --config-folder benchmark/config/large
```
