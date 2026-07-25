| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-2dc51d469cc3ca164adf825af0c27d076f3c7e8a.md) | 478 |  4,000,051 |  244 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-2dc51d469cc3ca164adf825af0c27d076f3c7e8a.md) | 10,281 |  14,365,133 |  1,547 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-2dc51d469cc3ca164adf825af0c27d076f3c7e8a.md) | 4,655 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-2dc51d469cc3ca164adf825af0c27d076f3c7e8a.md) | 683 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-2dc51d469cc3ca164adf825af0c27d076f3c7e8a.md) | 228 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-2dc51d469cc3ca164adf825af0c27d076f3c7e8a.md) | 274 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-2dc51d469cc3ca164adf825af0c27d076f3c7e8a.md) | 2,385 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2dc51d469cc3ca164adf825af0c27d076f3c7e8a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30143632800)
