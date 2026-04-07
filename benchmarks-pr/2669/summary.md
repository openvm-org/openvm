| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/fibonacci-0c7add92691384ea17b4829dde06aba7a9355d45.md) | 3,792 |  12,000,265 |  946 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/keccak-0c7add92691384ea17b4829dde06aba7a9355d45.md) | 18,461 |  18,655,329 |  3,305 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/regex-0c7add92691384ea17b4829dde06aba7a9355d45.md) | 1,418 |  4,137,067 |  372 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/ecrecover-0c7add92691384ea17b4829dde06aba7a9355d45.md) | 645 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/pairing-0c7add92691384ea17b4829dde06aba7a9355d45.md) | 902 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/kitchen_sink-0c7add92691384ea17b4829dde06aba7a9355d45.md) | 2,151 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0c7add92691384ea17b4829dde06aba7a9355d45

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24105288072)
