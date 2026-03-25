| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/fibonacci-2c5a7720e4bd56ff0bbc41ccbfbe480d29dbe810.md) | 4,155 |  12,000,265 |  1,355 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/keccak-2c5a7720e4bd56ff0bbc41ccbfbe480d29dbe810.md) | 19,328 |  1,235,218 |  3,395 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/regex-2c5a7720e4bd56ff0bbc41ccbfbe480d29dbe810.md) | 1,604 |  4,136,694 |  525 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/ecrecover-2c5a7720e4bd56ff0bbc41ccbfbe480d29dbe810.md) | 652 |  122,348 |  339 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/pairing-2c5a7720e4bd56ff0bbc41ccbfbe480d29dbe810.md) | 1,063 |  1,745,757 |  338 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/kitchen_sink-2c5a7720e4bd56ff0bbc41ccbfbe480d29dbe810.md) | 3,305 |  154,763 |  727 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2c5a7720e4bd56ff0bbc41ccbfbe480d29dbe810

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23567153090)
