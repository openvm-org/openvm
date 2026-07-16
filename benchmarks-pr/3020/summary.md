| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-1371c130b526bb0260a0a8cf8c31b3b9546c7959.md) | 475 |  4,000,051 |  238 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-1371c130b526bb0260a0a8cf8c31b3b9546c7959.md) | 7,168 |  14,365,133 |  1,534 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-1371c130b526bb0260a0a8cf8c31b3b9546c7959.md) | 4,455 |  11,167,961 |  533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-1371c130b526bb0260a0a8cf8c31b3b9546c7959.md) | 657 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-1371c130b526bb0260a0a8cf8c31b3b9546c7959.md) | 220 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-1371c130b526bb0260a0a8cf8c31b3b9546c7959.md) | 254 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-1371c130b526bb0260a0a8cf8c31b3b9546c7959.md) | 2,720 |  1,979,971 |  473 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1371c130b526bb0260a0a8cf8c31b3b9546c7959

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29474805535)
