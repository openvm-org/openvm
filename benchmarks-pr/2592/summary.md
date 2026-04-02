| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-acfcec5227642018a4635446b361a99935a08424.md) | 3,840 |  12,000,265 |  942 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-acfcec5227642018a4635446b361a99935a08424.md) | 18,674 |  18,655,329 |  3,333 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-acfcec5227642018a4635446b361a99935a08424.md) | 1,441 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-acfcec5227642018a4635446b361a99935a08424.md) | 649 |  123,583 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-acfcec5227642018a4635446b361a99935a08424.md) | 902 |  1,745,757 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-acfcec5227642018a4635446b361a99935a08424.md) | 2,265 |  2,579,903 |  441 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/acfcec5227642018a4635446b361a99935a08424

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23878343436)
