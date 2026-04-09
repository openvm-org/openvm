| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-2a4edb53dcf266766a27c819ccd977a1143282ee.md) | 3,830 |  12,000,265 |  948 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-2a4edb53dcf266766a27c819ccd977a1143282ee.md) | 18,752 |  18,655,329 |  3,353 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-2a4edb53dcf266766a27c819ccd977a1143282ee.md) | 1,428 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-2a4edb53dcf266766a27c819ccd977a1143282ee.md) | 648 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-2a4edb53dcf266766a27c819ccd977a1143282ee.md) | 910 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-2a4edb53dcf266766a27c819ccd977a1143282ee.md) | 2,169 |  2,579,903 |  444 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2a4edb53dcf266766a27c819ccd977a1143282ee

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24210612818)
