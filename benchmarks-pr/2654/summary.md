| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2654/fibonacci-8705a195d51114be4bd474cfaadf22e6f34e0fa6.md) | 3,778 |  12,000,265 |  933 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2654/keccak-8705a195d51114be4bd474cfaadf22e6f34e0fa6.md) | 18,527 |  18,655,329 |  3,334 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2654/regex-8705a195d51114be4bd474cfaadf22e6f34e0fa6.md) | 1,424 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2654/ecrecover-8705a195d51114be4bd474cfaadf22e6f34e0fa6.md) | 735 |  317,792 |  355 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2654/pairing-8705a195d51114be4bd474cfaadf22e6f34e0fa6.md) | 904 |  1,745,757 |  311 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2654/kitchen_sink-8705a195d51114be4bd474cfaadf22e6f34e0fa6.md) | 2,497 |  2,580,026 |  545 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8705a195d51114be4bd474cfaadf22e6f34e0fa6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23913378228)
