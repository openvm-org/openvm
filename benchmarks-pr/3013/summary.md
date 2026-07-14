| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/fibonacci-88f63f02b90bec58a9d8364029f139a8ff787888.md) |<span style='color: green'>(-2561 [-84.4%])</span> 473 | <span style='color: green'>(-8000214 [-66.7%])</span> 4,000,051 | <span style='color: green'>(-438 [-65.3%])</span> 233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/keccak-88f63f02b90bec58a9d8364029f139a8ff787888.md) |<span style='color: green'>(-7593 [-46.5%])</span> 8,736 | <span style='color: green'>(-4290196 [-23.0%])</span> 14,365,133 | <span style='color: green'>(-1474 [-48.7%])</span> 1,554 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/sha2_bench-88f63f02b90bec58a9d8364029f139a8ff787888.md) |<span style='color: green'>(-5057 [-55.3%])</span> 4,080 | <span style='color: green'>(-3625999 [-24.5%])</span> 11,167,961 | <span style='color: green'>(-595 [-53.0%])</span> 528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/regex-88f63f02b90bec58a9d8364029f139a8ff787888.md) |<span style='color: green'>(-597 [-51.2%])</span> 570 | <span style='color: green'>(-46411 [-1.1%])</span> 4,090,656 | <span style='color: green'>(-136 [-38.7%])</span> 215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/ecrecover-88f63f02b90bec58a9d8364029f139a8ff787888.md) |<span style='color: green'>(-378 [-63.2%])</span> 220 | <span style='color: green'>(-11373 [-9.2%])</span> 112,210 | <span style='color: green'>(-103 [-36.3%])</span> 181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/pairing-88f63f02b90bec58a9d8364029f139a8ff787888.md) |<span style='color: green'>(-652 [-70.0%])</span> 279 | <span style='color: green'>(-1152930 [-66.0%])</span> 592,827 | <span style='color: green'>(-124 [-40.3%])</span> 184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/kitchen_sink-88f63f02b90bec58a9d8364029f139a8ff787888.md) |<span style='color: green'>(-2176 [-52.8%])</span> 1,949 | <span style='color: green'>(-599932 [-23.3%])</span> 1,979,971 | <span style='color: green'>(-420 [-47.7%])</span> 460 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/fibonacci_e2e-88f63f02b90bec58a9d8364029f139a8ff787888.md) | 501 |  4,000,051 |  217 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/regex_e2e-88f63f02b90bec58a9d8364029f139a8ff787888.md) | 672 |  4,090,656 |  206 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/ecrecover_e2e-88f63f02b90bec58a9d8364029f139a8ff787888.md) | 214 |  112,210 |  173 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/pairing_e2e-88f63f02b90bec58a9d8364029f139a8ff787888.md) | 305 |  592,827 |  175 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/kitchen_sink_e2e-88f63f02b90bec58a9d8364029f139a8ff787888.md) | 2,296 |  1,979,971 |  454 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/88f63f02b90bec58a9d8364029f139a8ff787888

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29363404598)
