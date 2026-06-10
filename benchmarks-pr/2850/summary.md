| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/fibonacci-2ce0fa4690e620ba7617bb2338c1271104821a06.md) | 5,367 |  4,000,051 |  440 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/keccak-2ce0fa4690e620ba7617bb2338c1271104821a06.md) | 18,597 |  14,365,133 |  2,366 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/sha2_bench-2ce0fa4690e620ba7617bb2338c1271104821a06.md) | 12,515 |  11,167,961 |  1,405 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/regex-2ce0fa4690e620ba7617bb2338c1271104821a06.md) | 3,615 |  4,090,656 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/ecrecover-2ce0fa4690e620ba7617bb2338c1271104821a06.md) | 1,961 |  112,210 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/pairing-2ce0fa4690e620ba7617bb2338c1271104821a06.md) | 2,063 |  592,827 |  253 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/kitchen_sink-2ce0fa4690e620ba7617bb2338c1271104821a06.md) | 6,026 |  1,979,971 |  950 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2ce0fa4690e620ba7617bb2338c1271104821a06

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27301060177)
