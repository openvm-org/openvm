| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/fibonacci-f3348cfa34aa3a24f4243b04311291ef3dc183da.md) | 3,840 |  12,000,265 |  946 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/keccak-f3348cfa34aa3a24f4243b04311291ef3dc183da.md) | 15,748 |  1,235,218 |  2,223 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/regex-f3348cfa34aa3a24f4243b04311291ef3dc183da.md) | 1,416 |  4,136,694 |  368 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/ecrecover-f3348cfa34aa3a24f4243b04311291ef3dc183da.md) | 634 |  122,348 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/pairing-f3348cfa34aa3a24f4243b04311291ef3dc183da.md) | 916 |  1,745,757 |  278 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/kitchen_sink-f3348cfa34aa3a24f4243b04311291ef3dc183da.md) | 2,362 |  154,763 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f3348cfa34aa3a24f4243b04311291ef3dc183da

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23914219023)
