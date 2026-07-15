| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/fibonacci-5f59735ff6f5d1ed8d97c3d391b230fedd5b4b98.md) | 461 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/keccak-5f59735ff6f5d1ed8d97c3d391b230fedd5b4b98.md) | 8,774 |  14,365,133 |  1,533 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/sha2_bench-5f59735ff6f5d1ed8d97c3d391b230fedd5b4b98.md) | 3,904 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/regex-5f59735ff6f5d1ed8d97c3d391b230fedd5b4b98.md) | 506 |  4,090,656 |  192 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/ecrecover-5f59735ff6f5d1ed8d97c3d391b230fedd5b4b98.md) | 221 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/pairing-5f59735ff6f5d1ed8d97c3d391b230fedd5b4b98.md) | 263 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2983/kitchen_sink-5f59735ff6f5d1ed8d97c3d391b230fedd5b4b98.md) | 1,907 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5f59735ff6f5d1ed8d97c3d391b230fedd5b4b98

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29398501081)
