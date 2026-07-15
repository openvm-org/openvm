| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/fibonacci-4d5c05e23209fd18c22b1ec97583a8d9e06d8bae.md) |<span style='color: green'>(-2596 [-85.6%])</span> 438 | <span style='color: green'>(-8000214 [-66.7%])</span> 4,000,051 | <span style='color: green'>(-444 [-66.2%])</span> 227 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/keccak-4d5c05e23209fd18c22b1ec97583a8d9e06d8bae.md) |<span style='color: green'>(-7830 [-48.0%])</span> 8,499 | <span style='color: green'>(-4290196 [-23.0%])</span> 14,365,133 | <span style='color: green'>(-1485 [-49.0%])</span> 1,543 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/sha2_bench-4d5c05e23209fd18c22b1ec97583a8d9e06d8bae.md) |<span style='color: green'>(-5114 [-56.0%])</span> 4,023 | <span style='color: green'>(-3625999 [-24.5%])</span> 11,167,961 | <span style='color: green'>(-593 [-52.8%])</span> 530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/regex-4d5c05e23209fd18c22b1ec97583a8d9e06d8bae.md) |<span style='color: green'>(-588 [-50.4%])</span> 579 | <span style='color: green'>(-46411 [-1.1%])</span> 4,090,656 | <span style='color: green'>(-138 [-39.3%])</span> 213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/ecrecover-4d5c05e23209fd18c22b1ec97583a8d9e06d8bae.md) |<span style='color: green'>(-381 [-63.7%])</span> 217 | <span style='color: green'>(-11373 [-9.2%])</span> 112,210 | <span style='color: green'>(-101 [-35.6%])</span> 183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/pairing-4d5c05e23209fd18c22b1ec97583a8d9e06d8bae.md) |<span style='color: green'>(-663 [-71.2%])</span> 268 | <span style='color: green'>(-1152930 [-66.0%])</span> 592,827 | <span style='color: green'>(-125 [-40.6%])</span> 183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/kitchen_sink-4d5c05e23209fd18c22b1ec97583a8d9e06d8bae.md) |<span style='color: green'>(-2236 [-54.2%])</span> 1,889 | <span style='color: green'>(-599932 [-23.3%])</span> 1,979,971 | <span style='color: green'>(-420 [-47.7%])</span> 460 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/fibonacci_e2e-4d5c05e23209fd18c22b1ec97583a8d9e06d8bae.md) | 453 |  4,000,051 |  223 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/regex_e2e-4d5c05e23209fd18c22b1ec97583a8d9e06d8bae.md) | 586 |  4,090,656 |  203 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/ecrecover_e2e-4d5c05e23209fd18c22b1ec97583a8d9e06d8bae.md) | 219 |  112,210 |  175 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/pairing_e2e-4d5c05e23209fd18c22b1ec97583a8d9e06d8bae.md) | 271 |  592,827 |  175 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/kitchen_sink_e2e-4d5c05e23209fd18c22b1ec97583a8d9e06d8bae.md) | 2,273 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4d5c05e23209fd18c22b1ec97583a8d9e06d8bae

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29410610485)
