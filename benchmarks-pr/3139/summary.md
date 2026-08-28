| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/fibonacci-ffc2802b63a3f6fac972583d78d31923d92b4ae8.md) |<span style='color: green'>(-1208 [-71.3%])</span> 487 | <span style='color: green'>(-8000214 [-66.7%])</span> 4,000,051 | <span style='color: green'>(-138 [-37.1%])</span> 234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/keccak-ffc2802b63a3f6fac972583d78d31923d92b4ae8.md) |<span style='color: green'>(-1873 [-19.6%])</span> 7,665 | <span style='color: green'>(-4290196 [-23.0%])</span> 14,365,133 | <span style='color: red'>(+92 [+6.0%])</span> 1,637 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/sha2_bench-ffc2802b63a3f6fac972583d78d31923d92b4ae8.md) |<span style='color: green'>(-909 [-17.3%])</span> 4,334 | <span style='color: green'>(-3625999 [-24.5%])</span> 11,167,961 | <span style='color: green'>(-57 [-9.7%])</span> 529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/regex-ffc2802b63a3f6fac972583d78d31923d92b4ae8.md) |<span style='color: red'>(+42 [+5.9%])</span> 751 | <span style='color: green'>(-46411 [-1.1%])</span> 4,090,656 | <span style='color: green'>(-2 [-0.9%])</span> 217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/ecrecover-ffc2802b63a3f6fac972583d78d31923d92b4ae8.md) |<span style='color: green'>(-233 [-52.7%])</span> 209 | <span style='color: green'>(-11373 [-9.2%])</span> 112,210 | <span style='color: green'>(-2 [-1.1%])</span> 188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/pairing-ffc2802b63a3f6fac972583d78d31923d92b4ae8.md) |<span style='color: green'>(-339 [-57.5%])</span> 251 | <span style='color: green'>(-1152930 [-66.0%])</span> 592,827 | <span style='color: green'>(-23 [-11.7%])</span> 173 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/kitchen_sink-ffc2802b63a3f6fac972583d78d31923d92b4ae8.md) |<span style='color: green'>(-29 [-1.3%])</span> 2,261 | <span style='color: green'>(-599932 [-23.3%])</span> 1,979,971 | <span style='color: green'>(-16 [-3.2%])</span> 481 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/fibonacci_e2e-ffc2802b63a3f6fac972583d78d31923d92b4ae8.md) | 773 |  4,000,053 |  226 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/regex_e2e-ffc2802b63a3f6fac972583d78d31923d92b4ae8.md) | 1,092 |  4,090,658 |  208 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/ecrecover_e2e-ffc2802b63a3f6fac972583d78d31923d92b4ae8.md) | 508 |  112,212 |  176 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/pairing_e2e-ffc2802b63a3f6fac972583d78d31923d92b4ae8.md) | 558 |  592,829 |  163 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/kitchen_sink_e2e-ffc2802b63a3f6fac972583d78d31923d92b4ae8.md) | 2,484 |  1,979,973 |  471 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ffc2802b63a3f6fac972583d78d31923d92b4ae8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33195648040)
