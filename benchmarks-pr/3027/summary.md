| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/fibonacci-7104af885b6552e22ef38cb11e8618302714e8c8.md) |<span style='color: green'>(-1482 [-48.7%])</span> 1,561 |  12,000,265 | <span style='color: green'>(-317 [-47.0%])</span> 358 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/keccak-7104af885b6552e22ef38cb11e8618302714e8c8.md) |<span style='color: green'>(-7061 [-43.3%])</span> 9,228 |  18,655,329 | <span style='color: green'>(-1489 [-49.2%])</span> 1,538 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/sha2_bench-7104af885b6552e22ef38cb11e8618302714e8c8.md) |<span style='color: green'>(-4287 [-46.2%])</span> 4,983 |  14,793,960 | <span style='color: green'>(-554 [-48.9%])</span> 578 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/regex-7104af885b6552e22ef38cb11e8618302714e8c8.md) |<span style='color: green'>(-493 [-42.5%])</span> 668 |  4,137,067 | <span style='color: green'>(-140 [-39.8%])</span> 212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/ecrecover-7104af885b6552e22ef38cb11e8618302714e8c8.md) |<span style='color: green'>(-170 [-28.4%])</span> 429 |  123,583 | <span style='color: green'>(-94 [-33.3%])</span> 188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/pairing-7104af885b6552e22ef38cb11e8618302714e8c8.md) |<span style='color: green'>(-339 [-36.1%])</span> 599 |  1,745,757 | <span style='color: green'>(-116 [-37.5%])</span> 193 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/kitchen_sink-7104af885b6552e22ef38cb11e8618302714e8c8.md) |<span style='color: green'>(-1920 [-46.6%])</span> 2,197 |  2,579,903 | <span style='color: green'>(-407 [-46.3%])</span> 472 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/fibonacci_e2e-7104af885b6552e22ef38cb11e8618302714e8c8.md) | 1,620 |  12,000,265 |  350 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/regex_e2e-7104af885b6552e22ef38cb11e8618302714e8c8.md) | 764 |  4,137,067 |  200 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/ecrecover_e2e-7104af885b6552e22ef38cb11e8618302714e8c8.md) | 503 |  123,583 |  179 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/pairing_e2e-7104af885b6552e22ef38cb11e8618302714e8c8.md) | 662 |  1,745,757 |  180 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3027/kitchen_sink_e2e-7104af885b6552e22ef38cb11e8618302714e8c8.md) | 2,703 |  2,579,903 |  466 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7104af885b6552e22ef38cb11e8618302714e8c8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29452367110)
