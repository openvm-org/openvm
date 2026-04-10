| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-0d19e9c270d7fab26fc9532e4088126191dc94e8.md) | 3,832 |  12,000,265 |  958 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-0d19e9c270d7fab26fc9532e4088126191dc94e8.md) | 18,552 |  18,655,329 |  3,318 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-0d19e9c270d7fab26fc9532e4088126191dc94e8.md) | 1,412 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-0d19e9c270d7fab26fc9532e4088126191dc94e8.md) | 643 |  123,583 |  264 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-0d19e9c270d7fab26fc9532e4088126191dc94e8.md) | 915 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-0d19e9c270d7fab26fc9532e4088126191dc94e8.md) | 2,158 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0d19e9c270d7fab26fc9532e4088126191dc94e8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24256817161)
