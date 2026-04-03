| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-ba5727148c01b17b7220d7148e43460e65c59b8e.md) | 3,843 |  12,000,265 |  943 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-ba5727148c01b17b7220d7148e43460e65c59b8e.md) | 18,408 |  18,655,329 |  3,314 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-ba5727148c01b17b7220d7148e43460e65c59b8e.md) | 1,423 |  4,137,067 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-ba5727148c01b17b7220d7148e43460e65c59b8e.md) | 640 |  123,583 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-ba5727148c01b17b7220d7148e43460e65c59b8e.md) | 897 |  1,745,757 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-ba5727148c01b17b7220d7148e43460e65c59b8e.md) | 2,292 |  2,579,903 |  438 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ba5727148c01b17b7220d7148e43460e65c59b8e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23929275394)
