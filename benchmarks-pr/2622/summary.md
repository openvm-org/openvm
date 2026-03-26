| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/fibonacci-a105812fbf07fc92e9333ed2454d54599fc0331e.md) | 3,835 |  12,000,265 |  950 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/keccak-a105812fbf07fc92e9333ed2454d54599fc0331e.md) | 15,761 |  1,235,218 |  2,191 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/regex-a105812fbf07fc92e9333ed2454d54599fc0331e.md) | 1,411 |  4,136,694 |  368 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/ecrecover-a105812fbf07fc92e9333ed2454d54599fc0331e.md) | 637 |  122,348 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/pairing-a105812fbf07fc92e9333ed2454d54599fc0331e.md) | 917 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/kitchen_sink-a105812fbf07fc92e9333ed2454d54599fc0331e.md) | 2,366 |  154,763 |  404 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a105812fbf07fc92e9333ed2454d54599fc0331e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23596750457)
