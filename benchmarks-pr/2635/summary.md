| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2635/fibonacci-51d76b69147c79696d85b8aee9c30f9d5ecfdd03.md) | 3,832 |  12,000,265 |  953 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2635/keccak-51d76b69147c79696d85b8aee9c30f9d5ecfdd03.md) | 15,663 |  1,235,218 |  2,200 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2635/regex-51d76b69147c79696d85b8aee9c30f9d5ecfdd03.md) | 1,419 |  4,136,694 |  369 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2635/ecrecover-51d76b69147c79696d85b8aee9c30f9d5ecfdd03.md) | 642 |  122,348 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2635/pairing-51d76b69147c79696d85b8aee9c30f9d5ecfdd03.md) | 921 |  1,745,757 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2635/kitchen_sink-51d76b69147c79696d85b8aee9c30f9d5ecfdd03.md) | 2,380 |  154,763 |  415 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/51d76b69147c79696d85b8aee9c30f9d5ecfdd03

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23816271153)
