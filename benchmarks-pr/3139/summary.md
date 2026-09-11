| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/fibonacci-594b044e705891bcc2abd01578410da7dfcd1efe.md) |<span style='color: green'>(-1186 [-70.8%])</span> 489 | <span style='color: green'>(-8000214 [-66.7%])</span> 4,000,051 | <span style='color: green'>(-135 [-36.3%])</span> 237 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/keccak-594b044e705891bcc2abd01578410da7dfcd1efe.md) |<span style='color: green'>(-1678 [-17.7%])</span> 7,800 | <span style='color: green'>(-4290196 [-23.0%])</span> 14,365,133 | <span style='color: red'>(+108 [+7.0%])</span> 1,648 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/sha2_bench-594b044e705891bcc2abd01578410da7dfcd1efe.md) |<span style='color: green'>(-896 [-17.1%])</span> 4,347 | <span style='color: green'>(-3625999 [-24.5%])</span> 11,167,961 | <span style='color: green'>(-55 [-9.4%])</span> 533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/regex-594b044e705891bcc2abd01578410da7dfcd1efe.md) |<span style='color: red'>(+91 [+13.2%])</span> 783 | <span style='color: green'>(-46411 [-1.1%])</span> 4,090,656 | <span style='color: green'>(-1 [-0.5%])</span> 218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/ecrecover-594b044e705891bcc2abd01578410da7dfcd1efe.md) |<span style='color: green'>(-219 [-51.2%])</span> 209 | <span style='color: green'>(-11373 [-9.2%])</span> 112,210 | <span style='color: red'>(+1 [+0.5%])</span> 188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/pairing-594b044e705891bcc2abd01578410da7dfcd1efe.md) |<span style='color: green'>(-336 [-57.0%])</span> 253 | <span style='color: green'>(-1152930 [-66.0%])</span> 592,827 | <span style='color: green'>(-25 [-12.7%])</span> 172 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/kitchen_sink-594b044e705891bcc2abd01578410da7dfcd1efe.md) |<span style='color: green'>(-45 [-2.0%])</span> 2,243 | <span style='color: green'>(-599932 [-23.3%])</span> 1,979,971 | <span style='color: green'>(-23 [-4.6%])</span> 474 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/fibonacci_e2e-594b044e705891bcc2abd01578410da7dfcd1efe.md) | 765 |  4,000,053 |  223 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/regex_e2e-594b044e705891bcc2abd01578410da7dfcd1efe.md) | 1,035 |  4,090,658 |  205 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/ecrecover_e2e-594b044e705891bcc2abd01578410da7dfcd1efe.md) | 509 |  112,212 |  177 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/pairing_e2e-594b044e705891bcc2abd01578410da7dfcd1efe.md) | 547 |  592,829 |  161 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/kitchen_sink_e2e-594b044e705891bcc2abd01578410da7dfcd1efe.md) | 2,465 |  1,979,973 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/594b044e705891bcc2abd01578410da7dfcd1efe

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/34626814620)
