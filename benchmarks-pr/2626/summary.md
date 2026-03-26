| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2626/fibonacci-752c4a9486e3a4103d6b14f2e36490fb46295906.md) | 3,864 |  12,000,265 |  946 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2626/keccak-752c4a9486e3a4103d6b14f2e36490fb46295906.md) | 15,723 |  1,235,218 |  2,203 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2626/regex-752c4a9486e3a4103d6b14f2e36490fb46295906.md) | 1,415 |  4,136,694 |  368 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2626/ecrecover-752c4a9486e3a4103d6b14f2e36490fb46295906.md) | 635 |  122,348 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2626/pairing-752c4a9486e3a4103d6b14f2e36490fb46295906.md) | 928 |  1,745,757 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2626/kitchen_sink-752c4a9486e3a4103d6b14f2e36490fb46295906.md) | 2,382 |  154,763 |  414 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/752c4a9486e3a4103d6b14f2e36490fb46295906

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23622400485)
