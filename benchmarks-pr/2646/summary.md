| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/fibonacci-3898ae80507a26dc643a1711dac2698f3a25e6aa.md) | 3,836 |  12,000,265 |  953 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/keccak-3898ae80507a26dc643a1711dac2698f3a25e6aa.md) | 18,669 |  18,655,329 |  3,357 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/regex-3898ae80507a26dc643a1711dac2698f3a25e6aa.md) | 1,420 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/ecrecover-3898ae80507a26dc643a1711dac2698f3a25e6aa.md) | 746 |  317,792 |  359 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/pairing-3898ae80507a26dc643a1711dac2698f3a25e6aa.md) | 923 |  1,745,757 |  316 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/kitchen_sink-3898ae80507a26dc643a1711dac2698f3a25e6aa.md) | 2,355 |  2,580,026 |  776 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3898ae80507a26dc643a1711dac2698f3a25e6aa

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24257197495)
