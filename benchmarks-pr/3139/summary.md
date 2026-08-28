| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/fibonacci-a415dcfc3e175f3138e910452cff75a79515599a.md) |<span style='color: green'>(-1213 [-71.6%])</span> 482 | <span style='color: green'>(-8000214 [-66.7%])</span> 4,000,051 | <span style='color: green'>(-140 [-37.6%])</span> 232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/keccak-a415dcfc3e175f3138e910452cff75a79515599a.md) |<span style='color: green'>(-1772 [-18.6%])</span> 7,766 | <span style='color: green'>(-4290196 [-23.0%])</span> 14,365,133 | <span style='color: red'>(+102 [+6.6%])</span> 1,647 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/sha2_bench-a415dcfc3e175f3138e910452cff75a79515599a.md) |<span style='color: green'>(-836 [-15.9%])</span> 4,407 | <span style='color: green'>(-3625999 [-24.5%])</span> 11,167,961 | <span style='color: green'>(-54 [-9.2%])</span> 532 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/regex-a415dcfc3e175f3138e910452cff75a79515599a.md) |<span style='color: red'>(+56 [+7.9%])</span> 765 | <span style='color: green'>(-46411 [-1.1%])</span> 4,090,656 | <span style='color: green'>(-1 [-0.5%])</span> 218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/ecrecover-a415dcfc3e175f3138e910452cff75a79515599a.md) |<span style='color: green'>(-232 [-52.5%])</span> 210 | <span style='color: green'>(-11373 [-9.2%])</span> 112,210 |  190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/pairing-a415dcfc3e175f3138e910452cff75a79515599a.md) |<span style='color: green'>(-339 [-57.5%])</span> 251 | <span style='color: green'>(-1152930 [-66.0%])</span> 592,827 | <span style='color: green'>(-22 [-11.2%])</span> 174 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3139/kitchen_sink-a415dcfc3e175f3138e910452cff75a79515599a.md) |<span style='color: green'>(-46 [-2.0%])</span> 2,244 | <span style='color: green'>(-599932 [-23.3%])</span> 1,979,971 | <span style='color: green'>(-22 [-4.4%])</span> 475 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a415dcfc3e175f3138e910452cff75a79515599a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33191314171)
