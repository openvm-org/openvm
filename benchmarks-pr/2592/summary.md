| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-c6273164458da62c4505981e43c78fde6cc88ddd.md) | 3,836 |  12,000,265 |  961 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-c6273164458da62c4505981e43c78fde6cc88ddd.md) | 18,535 |  18,655,329 |  3,326 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-c6273164458da62c4505981e43c78fde6cc88ddd.md) | 1,394 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-c6273164458da62c4505981e43c78fde6cc88ddd.md) | 642 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-c6273164458da62c4505981e43c78fde6cc88ddd.md) | 911 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-c6273164458da62c4505981e43c78fde6cc88ddd.md) | 2,156 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c6273164458da62c4505981e43c78fde6cc88ddd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24253899222)
