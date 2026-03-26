| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-6f8531d17c14a9a4e318f91314e6cbfa3b47ae61.md) | 3,836 |  12,000,265 |  939 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-6f8531d17c14a9a4e318f91314e6cbfa3b47ae61.md) | 18,361 |  18,655,329 |  3,247 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-6f8531d17c14a9a4e318f91314e6cbfa3b47ae61.md) | 1,420 |  4,137,067 |  368 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-6f8531d17c14a9a4e318f91314e6cbfa3b47ae61.md) | 652 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-6f8531d17c14a9a4e318f91314e6cbfa3b47ae61.md) | 912 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-6f8531d17c14a9a4e318f91314e6cbfa3b47ae61.md) | 2,274 |  2,579,903 |  429 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6f8531d17c14a9a4e318f91314e6cbfa3b47ae61

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23616377946)
