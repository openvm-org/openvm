| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/fibonacci-ec8553cae9ce1195380d41d87a6061ba94faf3bf.md) |<span style='color: green'>(-2630 [-86.7%])</span> 404 | <span style='color: green'>(-8000214 [-66.7%])</span> 4,000,051 | <span style='color: green'>(-442 [-65.9%])</span> 229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/keccak-ec8553cae9ce1195380d41d87a6061ba94faf3bf.md) |<span style='color: green'>(-7959 [-48.7%])</span> 8,370 | <span style='color: green'>(-4290196 [-23.0%])</span> 14,365,133 | <span style='color: green'>(-1504 [-49.7%])</span> 1,524 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/sha2_bench-ec8553cae9ce1195380d41d87a6061ba94faf3bf.md) |<span style='color: green'>(-5194 [-56.8%])</span> 3,943 | <span style='color: green'>(-3625999 [-24.5%])</span> 11,167,961 | <span style='color: green'>(-600 [-53.4%])</span> 523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/regex-ec8553cae9ce1195380d41d87a6061ba94faf3bf.md) |<span style='color: green'>(-597 [-51.2%])</span> 570 | <span style='color: green'>(-46411 [-1.1%])</span> 4,090,656 | <span style='color: green'>(-139 [-39.6%])</span> 212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/ecrecover-ec8553cae9ce1195380d41d87a6061ba94faf3bf.md) |<span style='color: green'>(-374 [-62.5%])</span> 224 | <span style='color: green'>(-11373 [-9.2%])</span> 112,210 | <span style='color: green'>(-100 [-35.2%])</span> 184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/pairing-ec8553cae9ce1195380d41d87a6061ba94faf3bf.md) |<span style='color: green'>(-652 [-70.0%])</span> 279 | <span style='color: green'>(-1152930 [-66.0%])</span> 592,827 | <span style='color: green'>(-121 [-39.3%])</span> 187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/kitchen_sink-ec8553cae9ce1195380d41d87a6061ba94faf3bf.md) |<span style='color: green'>(-2209 [-53.6%])</span> 1,916 | <span style='color: green'>(-599932 [-23.3%])</span> 1,979,971 | <span style='color: green'>(-412 [-46.8%])</span> 468 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/fibonacci_e2e-ec8553cae9ce1195380d41d87a6061ba94faf3bf.md) | 430 |  4,000,051 |  223 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/regex_e2e-ec8553cae9ce1195380d41d87a6061ba94faf3bf.md) | 569 |  4,090,656 |  202 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/ecrecover_e2e-ec8553cae9ce1195380d41d87a6061ba94faf3bf.md) | 223 |  112,210 |  172 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/pairing_e2e-ec8553cae9ce1195380d41d87a6061ba94faf3bf.md) | 274 |  592,827 |  176 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3013/kitchen_sink_e2e-ec8553cae9ce1195380d41d87a6061ba94faf3bf.md) | 2,238 |  1,979,971 |  457 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ec8553cae9ce1195380d41d87a6061ba94faf3bf

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29415053783)
