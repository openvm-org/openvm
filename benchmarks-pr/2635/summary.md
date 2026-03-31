| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2635/fibonacci-c712bc1d5cdeee7c484305a38507f4975b18b0c1.md) | 3,868 |  12,000,265 |  966 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2635/keccak-c712bc1d5cdeee7c484305a38507f4975b18b0c1.md) | 15,729 |  1,235,218 |  2,197 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2635/regex-c712bc1d5cdeee7c484305a38507f4975b18b0c1.md) | 1,417 |  4,136,694 |  368 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2635/ecrecover-c712bc1d5cdeee7c484305a38507f4975b18b0c1.md) | 639 |  122,348 |  265 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2635/pairing-c712bc1d5cdeee7c484305a38507f4975b18b0c1.md) | 917 |  1,745,757 |  279 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2635/kitchen_sink-c712bc1d5cdeee7c484305a38507f4975b18b0c1.md) | 2,382 |  154,763 |  419 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c712bc1d5cdeee7c484305a38507f4975b18b0c1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23814288316)
