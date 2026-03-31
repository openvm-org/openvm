| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2632/fibonacci-32af2f2c9f3c40ffdfef4afda69a38f6c2874729.md) | 3,833 |  12,000,265 |  944 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2632/keccak-32af2f2c9f3c40ffdfef4afda69a38f6c2874729.md) | 18,707 |  18,655,329 |  3,330 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2632/regex-32af2f2c9f3c40ffdfef4afda69a38f6c2874729.md) | 1,410 |  4,137,067 |  367 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2632/ecrecover-32af2f2c9f3c40ffdfef4afda69a38f6c2874729.md) | 652 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2632/pairing-32af2f2c9f3c40ffdfef4afda69a38f6c2874729.md) | 938 |  1,745,757 |  292 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2632/kitchen_sink-32af2f2c9f3c40ffdfef4afda69a38f6c2874729.md) | 2,302 |  2,579,903 |  441 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/32af2f2c9f3c40ffdfef4afda69a38f6c2874729

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23810529601)
