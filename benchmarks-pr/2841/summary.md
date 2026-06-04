| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2841/fibonacci-e2b42d2c62b57396e52b7cc305a047f3b353889b.md) | 1,541 |  4,000,051 |  430 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2841/keccak-e2b42d2c62b57396e52b7cc305a047f3b353889b.md) | 13,898 |  14,365,133 |  2,377 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2841/sha2_bench-e2b42d2c62b57396e52b7cc305a047f3b353889b.md) | 9,008 |  11,167,961 |  1,430 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2841/regex-e2b42d2c62b57396e52b7cc305a047f3b353889b.md) | 1,474 |  4,090,656 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2841/ecrecover-e2b42d2c62b57396e52b7cc305a047f3b353889b.md) | 476 |  112,210 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2841/pairing-e2b42d2c62b57396e52b7cc305a047f3b353889b.md) | 612 |  592,827 |  259 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2841/kitchen_sink-e2b42d2c62b57396e52b7cc305a047f3b353889b.md) | 3,738 |  1,979,971 |  930 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e2b42d2c62b57396e52b7cc305a047f3b353889b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26968805620)
