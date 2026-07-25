| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-2dc51d469cc3ca164adf825af0c27d076f3c7e8a.md) | 479 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-2dc51d469cc3ca164adf825af0c27d076f3c7e8a.md) | 10,289 |  14,365,133 |  1,530 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-2dc51d469cc3ca164adf825af0c27d076f3c7e8a.md) | 4,652 |  11,167,961 |  527 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-2dc51d469cc3ca164adf825af0c27d076f3c7e8a.md) | 687 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-2dc51d469cc3ca164adf825af0c27d076f3c7e8a.md) | 228 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-2dc51d469cc3ca164adf825af0c27d076f3c7e8a.md) | 275 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-2dc51d469cc3ca164adf825af0c27d076f3c7e8a.md) | 2,377 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2dc51d469cc3ca164adf825af0c27d076f3c7e8a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30146546394)
