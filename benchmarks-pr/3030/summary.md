| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/fibonacci-893d9babc2db98ed14c918c70f7289abef4a2fa8.md) | 1,573 |  12,000,265 | <span style='color: green'>(-5 [-1.4%])</span> 358 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/keccak-893d9babc2db98ed14c918c70f7289abef4a2fa8.md) |<span style='color: red'>(+116 [+1.3%])</span> 9,390 |  18,655,329 | <span style='color: red'>(+9 [+0.6%])</span> 1,523 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/sha2_bench-893d9babc2db98ed14c918c70f7289abef4a2fa8.md) |<span style='color: green'>(-28 [-0.6%])</span> 4,919 |  14,793,960 | <span style='color: red'>(+2 [+0.3%])</span> 578 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/regex-893d9babc2db98ed14c918c70f7289abef4a2fa8.md) |<span style='color: green'>(-1 [-0.2%])</span> 660 |  4,137,067 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/ecrecover-893d9babc2db98ed14c918c70f7289abef4a2fa8.md) |<span style='color: red'>(+4 [+0.9%])</span> 435 |  123,583 | <span style='color: red'>(+8 [+4.4%])</span> 190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/pairing-893d9babc2db98ed14c918c70f7289abef4a2fa8.md) |<span style='color: green'>(-10 [-1.8%])</span> 559 |  1,745,757 |  190 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/kitchen_sink-893d9babc2db98ed14c918c70f7289abef4a2fa8.md) |<span style='color: red'>(+4 [+0.2%])</span> 2,207 |  2,579,903 |  475 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/893d9babc2db98ed14c918c70f7289abef4a2fa8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30290914640)
