| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3110/fibonacci-1a239f1f35208248e0bc66bc572b4b4813ac32b3.md) | 470 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3110/keccak-1a239f1f35208248e0bc66bc572b4b4813ac32b3.md) | 7,337 |  14,365,133 |  1,523 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3110/sha2_bench-1a239f1f35208248e0bc66bc572b4b4813ac32b3.md) | 4,081 |  11,167,961 |  516 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3110/regex-1a239f1f35208248e0bc66bc572b4b4813ac32b3.md) | 668 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3110/ecrecover-1a239f1f35208248e0bc66bc572b4b4813ac32b3.md) | 222 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3110/pairing-1a239f1f35208248e0bc66bc572b4b4813ac32b3.md) | 237 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3110/kitchen_sink-1a239f1f35208248e0bc66bc572b4b4813ac32b3.md) | 2,016 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1a239f1f35208248e0bc66bc572b4b4813ac32b3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31040758200)
