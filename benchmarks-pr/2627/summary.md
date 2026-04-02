| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/fibonacci-6fa909e0d622053361223ed720f282ef95d0cf35.md) | 3,825 |  12,000,265 |  941 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/keccak-6fa909e0d622053361223ed720f282ef95d0cf35.md) | 15,675 |  1,235,218 |  2,209 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/regex-6fa909e0d622053361223ed720f282ef95d0cf35.md) | 1,405 |  4,136,694 |  368 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/ecrecover-6fa909e0d622053361223ed720f282ef95d0cf35.md) | 632 |  122,348 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/pairing-6fa909e0d622053361223ed720f282ef95d0cf35.md) | 915 |  1,745,757 |  278 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/kitchen_sink-6fa909e0d622053361223ed720f282ef95d0cf35.md) | 2,395 |  154,763 |  417 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6fa909e0d622053361223ed720f282ef95d0cf35

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23877017335)
