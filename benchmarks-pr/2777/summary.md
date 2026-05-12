| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/fibonacci-535d53b0e9ec25e207c2717edc1123c639c9825e.md) | 1,884 |  4,000,051 |  535 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/keccak-535d53b0e9ec25e207c2717edc1123c639c9825e.md) | 13,594 |  14,365,133 |  2,238 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/sha2_bench-535d53b0e9ec25e207c2717edc1123c639c9825e.md) | 9,502 |  11,167,961 |  1,419 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/regex-535d53b0e9ec25e207c2717edc1123c639c9825e.md) | 1,594 |  4,090,656 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/ecrecover-535d53b0e9ec25e207c2717edc1123c639c9825e.md) | 636 |  112,210 |  293 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/pairing-535d53b0e9ec25e207c2717edc1123c639c9825e.md) | 756 |  592,827 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/kitchen_sink-535d53b0e9ec25e207c2717edc1123c639c9825e.md) | 2,035 |  1,979,971 |  429 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/535d53b0e9ec25e207c2717edc1123c639c9825e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25758906065)
