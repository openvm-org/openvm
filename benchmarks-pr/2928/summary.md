| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2928/fibonacci-8e6d552a4bcf767a6bebfcd893aec187447e72ac.md) | 1,025 |  4,000,051 |  387 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2928/keccak-8e6d552a4bcf767a6bebfcd893aec187447e72ac.md) | 16,440 |  14,365,133 |  3,034 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2928/sha2_bench-8e6d552a4bcf767a6bebfcd893aec187447e72ac.md) | 8,245 |  11,167,961 |  1,009 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2928/regex-8e6d552a4bcf767a6bebfcd893aec187447e72ac.md) | 1,228 |  4,090,656 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2928/ecrecover-8e6d552a4bcf767a6bebfcd893aec187447e72ac.md) | 437 |  112,210 |  275 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2928/pairing-8e6d552a4bcf767a6bebfcd893aec187447e72ac.md) | 603 |  592,827 |  293 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2928/kitchen_sink-8e6d552a4bcf767a6bebfcd893aec187447e72ac.md) | 3,910 |  1,979,971 |  863 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8e6d552a4bcf767a6bebfcd893aec187447e72ac

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28116345197)
