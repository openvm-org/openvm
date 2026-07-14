| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/fibonacci-5acc51a70bcc6465a1568e18b685b8e26451e190.md) |<span style='color: green'>(-2562 [-84.4%])</span> 472 | <span style='color: green'>(-8000214 [-66.7%])</span> 4,000,051 | <span style='color: green'>(-442 [-65.9%])</span> 229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/keccak-5acc51a70bcc6465a1568e18b685b8e26451e190.md) |<span style='color: green'>(-7467 [-45.7%])</span> 8,862 | <span style='color: green'>(-4290196 [-23.0%])</span> 14,365,133 | <span style='color: green'>(-1488 [-49.1%])</span> 1,540 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/sha2_bench-5acc51a70bcc6465a1568e18b685b8e26451e190.md) |<span style='color: green'>(-5180 [-56.7%])</span> 3,957 | <span style='color: green'>(-3625999 [-24.5%])</span> 11,167,961 | <span style='color: green'>(-603 [-53.7%])</span> 520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/regex-5acc51a70bcc6465a1568e18b685b8e26451e190.md) |<span style='color: green'>(-668 [-57.2%])</span> 499 | <span style='color: green'>(-46411 [-1.1%])</span> 4,090,656 | <span style='color: green'>(-157 [-44.7%])</span> 194 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/ecrecover-5acc51a70bcc6465a1568e18b685b8e26451e190.md) |<span style='color: green'>(-380 [-63.5%])</span> 218 | <span style='color: green'>(-11373 [-9.2%])</span> 112,210 | <span style='color: green'>(-101 [-35.6%])</span> 183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/pairing-5acc51a70bcc6465a1568e18b685b8e26451e190.md) |<span style='color: green'>(-668 [-71.8%])</span> 263 | <span style='color: green'>(-1152930 [-66.0%])</span> 592,827 | <span style='color: green'>(-126 [-40.9%])</span> 182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/kitchen_sink-5acc51a70bcc6465a1568e18b685b8e26451e190.md) |<span style='color: green'>(-2209 [-53.6%])</span> 1,916 | <span style='color: green'>(-599932 [-23.3%])</span> 1,979,971 | <span style='color: green'>(-415 [-47.2%])</span> 465 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/fibonacci_e2e-5acc51a70bcc6465a1568e18b685b8e26451e190.md) | 480 |  4,000,051 |  220 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/regex_e2e-5acc51a70bcc6465a1568e18b685b8e26451e190.md) | 587 |  4,090,656 |  178 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/ecrecover_e2e-5acc51a70bcc6465a1568e18b685b8e26451e190.md) | 214 |  112,210 |  175 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/pairing_e2e-5acc51a70bcc6465a1568e18b685b8e26451e190.md) | 283 |  592,827 |  177 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/kitchen_sink_e2e-5acc51a70bcc6465a1568e18b685b8e26451e190.md) | 2,288 |  1,979,971 |  457 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5acc51a70bcc6465a1568e18b685b8e26451e190

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29366536673)
