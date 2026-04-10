| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2692/fibonacci-ff298842825095d0a2c7449bd6b8d86bc7ef72be.md) | 3,781 |  12,000,265 |  946 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2692/keccak-ff298842825095d0a2c7449bd6b8d86bc7ef72be.md) | 18,377 |  18,655,329 |  3,281 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2692/regex-ff298842825095d0a2c7449bd6b8d86bc7ef72be.md) | 1,430 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2692/ecrecover-ff298842825095d0a2c7449bd6b8d86bc7ef72be.md) | 643 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2692/pairing-ff298842825095d0a2c7449bd6b8d86bc7ef72be.md) | 905 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2692/kitchen_sink-ff298842825095d0a2c7449bd6b8d86bc7ef72be.md) | 2,154 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ff298842825095d0a2c7449bd6b8d86bc7ef72be

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24243006527)
