| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2661/fibonacci-f247456c1120812b9a8b5e7958a6e8bcc4d2c513.md) | 3,827 |  12,000,265 |  960 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2661/keccak-f247456c1120812b9a8b5e7958a6e8bcc4d2c513.md) | 18,978 |  18,655,329 |  3,398 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2661/regex-f247456c1120812b9a8b5e7958a6e8bcc4d2c513.md) | 1,447 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2661/ecrecover-f247456c1120812b9a8b5e7958a6e8bcc4d2c513.md) | 651 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2661/pairing-f247456c1120812b9a8b5e7958a6e8bcc4d2c513.md) | 913 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2661/kitchen_sink-f247456c1120812b9a8b5e7958a6e8bcc4d2c513.md) | 2,274 |  2,579,903 |  442 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f247456c1120812b9a8b5e7958a6e8bcc4d2c513

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23962645435)
