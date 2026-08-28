| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/fibonacci-967f448e10194a88641e84b6fd4bdffd39639571.md) |<span style='color: green'>(-1194 [-70.4%])</span> 501 | <span style='color: green'>(-8000214 [-66.7%])</span> 4,000,051 | <span style='color: green'>(-134 [-36.0%])</span> 238 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/keccak-967f448e10194a88641e84b6fd4bdffd39639571.md) |<span style='color: green'>(-1890 [-19.8%])</span> 7,648 | <span style='color: green'>(-4290196 [-23.0%])</span> 14,365,133 | <span style='color: red'>(+57 [+3.7%])</span> 1,602 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/sha2_bench-967f448e10194a88641e84b6fd4bdffd39639571.md) |<span style='color: green'>(-828 [-15.8%])</span> 4,415 | <span style='color: green'>(-3625999 [-24.5%])</span> 11,167,961 | <span style='color: green'>(-52 [-8.9%])</span> 534 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/regex-967f448e10194a88641e84b6fd4bdffd39639571.md) |<span style='color: red'>(+59 [+8.3%])</span> 768 | <span style='color: green'>(-46411 [-1.1%])</span> 4,090,656 | <span style='color: green'>(-3 [-1.4%])</span> 216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/ecrecover-967f448e10194a88641e84b6fd4bdffd39639571.md) |<span style='color: green'>(-233 [-52.7%])</span> 209 | <span style='color: green'>(-11373 [-9.2%])</span> 112,210 |  190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/pairing-967f448e10194a88641e84b6fd4bdffd39639571.md) |<span style='color: green'>(-342 [-58.0%])</span> 248 | <span style='color: green'>(-1152930 [-66.0%])</span> 592,827 | <span style='color: green'>(-23 [-11.7%])</span> 173 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/kitchen_sink-967f448e10194a88641e84b6fd4bdffd39639571.md) |<span style='color: green'>(-45 [-2.0%])</span> 2,245 | <span style='color: green'>(-599932 [-23.3%])</span> 1,979,971 | <span style='color: green'>(-22 [-4.4%])</span> 475 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/fibonacci_e2e-967f448e10194a88641e84b6fd4bdffd39639571.md) | 811 |  4,000,053 |  225 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/regex_e2e-967f448e10194a88641e84b6fd4bdffd39639571.md) | 1,069 |  4,090,658 |  209 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/ecrecover_e2e-967f448e10194a88641e84b6fd4bdffd39639571.md) | 513 |  112,212 |  179 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/pairing_e2e-967f448e10194a88641e84b6fd4bdffd39639571.md) | 558 |  592,829 |  163 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/kitchen_sink_e2e-967f448e10194a88641e84b6fd4bdffd39639571.md) | 2,480 |  1,979,973 |  470 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/967f448e10194a88641e84b6fd4bdffd39639571

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33201448970)
