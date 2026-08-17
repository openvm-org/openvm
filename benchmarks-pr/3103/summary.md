| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-c859c16ab1f136897b48bed539dacf1fb1fffe91.md) | 472 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-c859c16ab1f136897b48bed539dacf1fb1fffe91.md) | 7,457 |  14,365,133 |  1,541 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-c859c16ab1f136897b48bed539dacf1fb1fffe91.md) | 4,111 |  11,167,961 |  516 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-c859c16ab1f136897b48bed539dacf1fb1fffe91.md) | 673 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-c859c16ab1f136897b48bed539dacf1fb1fffe91.md) | 196 |  112,210 |  200 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-c859c16ab1f136897b48bed539dacf1fb1fffe91.md) | 237 |  592,827 |  197 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-c859c16ab1f136897b48bed539dacf1fb1fffe91.md) | 2,018 |  1,979,971 |  520 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c859c16ab1f136897b48bed539dacf1fb1fffe91

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32066940509)
