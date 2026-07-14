| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/fibonacci-987f1fb4daebdebecc4e634204e0d3934cea6990.md) |<span style='color: green'>(-2168 [-71.5%])</span> 866 | <span style='color: green'>(-8000214 [-66.7%])</span> 4,000,051 | <span style='color: green'>(-283 [-42.2%])</span> 388 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/keccak-987f1fb4daebdebecc4e634204e0d3934cea6990.md) |<span style='color: green'>(-909 [-5.6%])</span> 15,420 | <span style='color: green'>(-4290196 [-23.0%])</span> 14,365,133 | <span style='color: red'>(+18 [+0.6%])</span> 3,046 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/sha2_bench-987f1fb4daebdebecc4e634204e0d3934cea6990.md) |<span style='color: green'>(-1140 [-12.5%])</span> 7,997 | <span style='color: green'>(-3625999 [-24.5%])</span> 11,167,961 | <span style='color: green'>(-119 [-10.6%])</span> 1,004 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/regex-987f1fb4daebdebecc4e634204e0d3934cea6990.md) |<span style='color: green'>(-136 [-11.7%])</span> 1,031 | <span style='color: green'>(-46411 [-1.1%])</span> 4,090,656 | <span style='color: red'>(+3 [+0.9%])</span> 354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/ecrecover-987f1fb4daebdebecc4e634204e0d3934cea6990.md) |<span style='color: green'>(-297 [-49.7%])</span> 301 | <span style='color: green'>(-11373 [-9.2%])</span> 112,210 | <span style='color: green'>(-6 [-2.1%])</span> 278 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/pairing-987f1fb4daebdebecc4e634204e0d3934cea6990.md) |<span style='color: green'>(-476 [-51.1%])</span> 455 | <span style='color: green'>(-1152930 [-66.0%])</span> 592,827 | <span style='color: green'>(-8 [-2.6%])</span> 300 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/kitchen_sink-987f1fb4daebdebecc4e634204e0d3934cea6990.md) |<span style='color: green'>(-400 [-9.7%])</span> 3,725 | <span style='color: green'>(-599932 [-23.3%])</span> 1,979,971 | <span style='color: green'>(-21 [-2.4%])</span> 859 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/fibonacci_e2e-987f1fb4daebdebecc4e634204e0d3934cea6990.md) | 411 |  4,000,051 |  178 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/regex_e2e-987f1fb4daebdebecc4e634204e0d3934cea6990.md) | 562 |  4,090,656 |  167 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/ecrecover_e2e-987f1fb4daebdebecc4e634204e0d3934cea6990.md) | 181 |  112,210 |  142 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/pairing_e2e-987f1fb4daebdebecc4e634204e0d3934cea6990.md) | 267 |  592,827 |  142 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/kitchen_sink_e2e-987f1fb4daebdebecc4e634204e0d3934cea6990.md) | 1,932 |  1,979,971 |  370 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/987f1fb4daebdebecc4e634204e0d3934cea6990

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29313052571)
