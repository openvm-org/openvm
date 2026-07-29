| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/fibonacci-97ef510a73a7f1ce83726522f27c30cfbd5b1943.md) | 457 |  4,000,051 |  239 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/keccak-97ef510a73a7f1ce83726522f27c30cfbd5b1943.md) | 7,229 |  14,365,133 |  1,520 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/sha2_bench-97ef510a73a7f1ce83726522f27c30cfbd5b1943.md) | 4,707 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/regex-97ef510a73a7f1ce83726522f27c30cfbd5b1943.md) | 650 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/ecrecover-97ef510a73a7f1ce83726522f27c30cfbd5b1943.md) | 225 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/pairing-97ef510a73a7f1ce83726522f27c30cfbd5b1943.md) | 298 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/kitchen_sink-97ef510a73a7f1ce83726522f27c30cfbd5b1943.md) | 2,655 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/97ef510a73a7f1ce83726522f27c30cfbd5b1943

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30432521824)
