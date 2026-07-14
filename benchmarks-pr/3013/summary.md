| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/fibonacci-cd67bab36b6208cdb278f36c69d1a70a44aeff93.md) |<span style='color: green'>(-2572 [-84.8%])</span> 462 | <span style='color: green'>(-8000214 [-66.7%])</span> 4,000,051 | <span style='color: green'>(-443 [-66.0%])</span> 228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/keccak-cd67bab36b6208cdb278f36c69d1a70a44aeff93.md) |<span style='color: green'>(-7765 [-47.6%])</span> 8,564 | <span style='color: green'>(-4290196 [-23.0%])</span> 14,365,133 | <span style='color: green'>(-1516 [-50.1%])</span> 1,512 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/sha2_bench-cd67bab36b6208cdb278f36c69d1a70a44aeff93.md) |<span style='color: green'>(-5051 [-55.3%])</span> 4,086 | <span style='color: green'>(-3625999 [-24.5%])</span> 11,167,961 | <span style='color: green'>(-599 [-53.3%])</span> 524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/regex-cd67bab36b6208cdb278f36c69d1a70a44aeff93.md) |<span style='color: green'>(-608 [-52.1%])</span> 559 | <span style='color: green'>(-46411 [-1.1%])</span> 4,090,656 | <span style='color: green'>(-138 [-39.3%])</span> 213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/ecrecover-cd67bab36b6208cdb278f36c69d1a70a44aeff93.md) |<span style='color: green'>(-382 [-63.9%])</span> 216 | <span style='color: green'>(-11373 [-9.2%])</span> 112,210 | <span style='color: green'>(-103 [-36.3%])</span> 181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/pairing-cd67bab36b6208cdb278f36c69d1a70a44aeff93.md) |<span style='color: green'>(-651 [-69.9%])</span> 280 | <span style='color: green'>(-1152930 [-66.0%])</span> 592,827 | <span style='color: green'>(-119 [-38.6%])</span> 189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/kitchen_sink-cd67bab36b6208cdb278f36c69d1a70a44aeff93.md) |<span style='color: green'>(-2173 [-52.7%])</span> 1,952 | <span style='color: green'>(-599932 [-23.3%])</span> 1,979,971 | <span style='color: green'>(-418 [-47.5%])</span> 462 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/fibonacci_e2e-cd67bab36b6208cdb278f36c69d1a70a44aeff93.md) | 477 |  4,000,051 |  217 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/regex_e2e-cd67bab36b6208cdb278f36c69d1a70a44aeff93.md) | 677 |  4,090,656 |  210 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/ecrecover_e2e-cd67bab36b6208cdb278f36c69d1a70a44aeff93.md) | 214 |  112,210 |  172 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/pairing_e2e-cd67bab36b6208cdb278f36c69d1a70a44aeff93.md) | 312 |  592,827 |  176 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/kitchen_sink_e2e-cd67bab36b6208cdb278f36c69d1a70a44aeff93.md) | 2,302 |  1,979,971 |  455 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/cd67bab36b6208cdb278f36c69d1a70a44aeff93

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29358559474)
