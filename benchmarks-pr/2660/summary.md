| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2660/fibonacci-41ee564bf356d7fe7635f05e946b839d4422d2b2.md) | 3,833 |  12,000,265 |  955 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2660/keccak-41ee564bf356d7fe7635f05e946b839d4422d2b2.md) | 15,708 |  1,235,218 |  2,205 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2660/regex-41ee564bf356d7fe7635f05e946b839d4422d2b2.md) | 1,435 |  4,136,694 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2660/ecrecover-41ee564bf356d7fe7635f05e946b839d4422d2b2.md) | 638 |  122,348 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2660/pairing-41ee564bf356d7fe7635f05e946b839d4422d2b2.md) | 924 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2660/kitchen_sink-41ee564bf356d7fe7635f05e946b839d4422d2b2.md) | 2,388 |  154,763 |  421 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/41ee564bf356d7fe7635f05e946b839d4422d2b2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23960288648)
