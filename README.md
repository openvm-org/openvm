# AFS

AFS stands for Axiom Flash Storage, or potentially Axiom Filesystem.

## Introduction to Plonky3

For a guide on plonky3, see [Getting Started](https://hackmd.io/@axiom/HJks1ZLGR).

## Benchmarks

To run benchmarks, install python3 and run:

```bash
python sdk/scripts/bench.py <name>
```

where `<name>` is a benchmark implemented as a rust binary (located in `src/bin` in a crate). Current benchmark options are:

- `verify_fibair`
- `tiny_e2e`
- `small_e2e`
  in the `recursion` crate.
  The benchmark outputs a JSON of metrics. You can process this into markdown with:

```bash
python sdk/scripts/metric_unify/main.py <path to json>
```

Currently the processing is done automatically at the end of `bench.py`. The script automatically detects if you have a previously saved metric file for the same benchmark and includes the diff report in the output.

Latest benchmark results can be found [here](https://github.com/axiom-crypto/afs-prototype/blob/gh-pages/index.md).
These are run via [github workflows](./.github/workflows/benchmark-call.yml) and should always be up to date with the latest `main` branch.
