| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2686/fibonacci-fc4a8794a8088d5abf6dd3560d97b4489f392548.md) | 3,804 |  12,000,265 |  945 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2686/keccak-fc4a8794a8088d5abf6dd3560d97b4489f392548.md) | 18,508 |  18,655,329 |  3,332 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2686/regex-fc4a8794a8088d5abf6dd3560d97b4489f392548.md) | 1,419 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2686/ecrecover-fc4a8794a8088d5abf6dd3560d97b4489f392548.md) | 651 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2686/pairing-fc4a8794a8088d5abf6dd3560d97b4489f392548.md) | 902 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2686/kitchen_sink-fc4a8794a8088d5abf6dd3560d97b4489f392548.md) | 2,152 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fc4a8794a8088d5abf6dd3560d97b4489f392548

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24209811934)
