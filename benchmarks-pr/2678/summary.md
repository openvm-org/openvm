| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/fibonacci-8c409cd120dbae1e3e6e2b2742f13c058f6f2282.md) | 3,911 |  12,000,265 |  987 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/keccak-8c409cd120dbae1e3e6e2b2742f13c058f6f2282.md) | 18,576 |  18,655,329 |  3,313 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/regex-8c409cd120dbae1e3e6e2b2742f13c058f6f2282.md) | 1,409 |  4,137,067 |  372 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/ecrecover-8c409cd120dbae1e3e6e2b2742f13c058f6f2282.md) | 665 |  123,583 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/pairing-8c409cd120dbae1e3e6e2b2742f13c058f6f2282.md) | 913 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/kitchen_sink-8c409cd120dbae1e3e6e2b2742f13c058f6f2282.md) | 2,165 |  2,579,903 |  435 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8c409cd120dbae1e3e6e2b2742f13c058f6f2282

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24226576072)
