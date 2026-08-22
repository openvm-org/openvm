| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3129/fibonacci-b0471ad23a281adb5fc1e2acebf5cd0c57e2181d.md) | 476 |  4,000,051 |  238 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3129/keccak-b0471ad23a281adb5fc1e2acebf5cd0c57e2181d.md) | 7,656 |  14,365,133 |  1,638 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3129/sha2_bench-b0471ad23a281adb5fc1e2acebf5cd0c57e2181d.md) | 4,321 |  11,167,961 |  528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3129/regex-b0471ad23a281adb5fc1e2acebf5cd0c57e2181d.md) | 764 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3129/ecrecover-b0471ad23a281adb5fc1e2acebf5cd0c57e2181d.md) | 209 |  112,210 |  191 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3129/pairing-b0471ad23a281adb5fc1e2acebf5cd0c57e2181d.md) | 246 |  592,827 |  193 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3129/kitchen_sink-b0471ad23a281adb5fc1e2acebf5cd0c57e2181d.md) | 2,229 |  1,979,971 |  471 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b0471ad23a281adb5fc1e2acebf5cd0c57e2181d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32563595377)
