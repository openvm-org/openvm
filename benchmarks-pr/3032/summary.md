| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/fibonacci-787fadd92cfd0456814153c74049b9792918d69d.md) |<span style='color: green'>(-15 [-0.9%])</span> 1,589 |  12,000,265 | <span style='color: green'>(-4 [-1.1%])</span> 360 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/keccak-787fadd92cfd0456814153c74049b9792918d69d.md) |<span style='color: green'>(-220 [-2.3%])</span> 9,184 |  18,655,329 | <span style='color: green'>(-36 [-2.3%])</span> 1,510 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/sha2_bench-787fadd92cfd0456814153c74049b9792918d69d.md) |<span style='color: red'>(+100 [+2.1%])</span> 4,954 |  14,793,960 | <span style='color: red'>(+6 [+1.0%])</span> 580 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/regex-787fadd92cfd0456814153c74049b9792918d69d.md) |<span style='color: red'>(+11 [+1.7%])</span> 664 |  4,137,067 | <span style='color: red'>(+1 [+0.5%])</span> 214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/ecrecover-787fadd92cfd0456814153c74049b9792918d69d.md) | 435 |  123,583 | <span style='color: red'>(+7 [+3.8%])</span> 192 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/pairing-787fadd92cfd0456814153c74049b9792918d69d.md) |<span style='color: green'>(-30 [-5.0%])</span> 568 |  1,745,757 | <span style='color: green'>(-1 [-0.5%])</span> 190 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/kitchen_sink-787fadd92cfd0456814153c74049b9792918d69d.md) |<span style='color: green'>(-6 [-0.3%])</span> 2,211 |  2,579,903 | <span style='color: red'>(+1 [+0.2%])</span> 480 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/fibonacci_e2e-787fadd92cfd0456814153c74049b9792918d69d.md) | 1,627 |  12,000,265 |  352 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/regex_e2e-787fadd92cfd0456814153c74049b9792918d69d.md) | 821 |  4,137,067 |  204 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/ecrecover_e2e-787fadd92cfd0456814153c74049b9792918d69d.md) | 505 |  123,583 |  176 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/pairing_e2e-787fadd92cfd0456814153c74049b9792918d69d.md) | 652 |  1,745,757 |  180 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3032/kitchen_sink_e2e-787fadd92cfd0456814153c74049b9792918d69d.md) | 2,699 |  2,579,903 |  469 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/787fadd92cfd0456814153c74049b9792918d69d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29596837089)
