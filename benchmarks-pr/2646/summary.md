| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/fibonacci-c00c20c787a62472f48eeba5ecb406b1d8643cd0.md) | 3,783 |  12,000,265 |  942 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/keccak-c00c20c787a62472f48eeba5ecb406b1d8643cd0.md) | 18,772 |  18,655,329 |  3,362 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/regex-c00c20c787a62472f48eeba5ecb406b1d8643cd0.md) | 1,412 |  4,137,067 |  379 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/ecrecover-c00c20c787a62472f48eeba5ecb406b1d8643cd0.md) | 743 |  317,792 |  358 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/pairing-c00c20c787a62472f48eeba5ecb406b1d8643cd0.md) | 909 |  1,745,757 |  315 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/kitchen_sink-c00c20c787a62472f48eeba5ecb406b1d8643cd0.md) | 2,378 |  2,580,026 |  790 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c00c20c787a62472f48eeba5ecb406b1d8643cd0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24160314477)
