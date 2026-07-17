| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/fibonacci-11e3769e6405ac6cc10b9dc0ea2ac64898df3287.md) |<span style='color: green'>(-45 [-2.8%])</span> 1,559 |  12,000,265 | <span style='color: green'>(-6 [-1.6%])</span> 358 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/keccak-11e3769e6405ac6cc10b9dc0ea2ac64898df3287.md) |<span style='color: green'>(-172 [-1.8%])</span> 9,232 |  18,655,329 | <span style='color: green'>(-35 [-2.3%])</span> 1,511 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/sha2_bench-11e3769e6405ac6cc10b9dc0ea2ac64898df3287.md) | 4,855 |  14,793,960 | <span style='color: green'>(-3 [-0.5%])</span> 571 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/regex-11e3769e6405ac6cc10b9dc0ea2ac64898df3287.md) |<span style='color: green'>(-3 [-0.5%])</span> 650 |  4,137,067 | <span style='color: red'>(+1 [+0.5%])</span> 214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/ecrecover-11e3769e6405ac6cc10b9dc0ea2ac64898df3287.md) |<span style='color: red'>(+3 [+0.7%])</span> 438 |  123,583 | <span style='color: green'>(-1 [-0.5%])</span> 184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/pairing-11e3769e6405ac6cc10b9dc0ea2ac64898df3287.md) |<span style='color: green'>(-46 [-7.7%])</span> 552 |  1,745,757 |  191 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/kitchen_sink-11e3769e6405ac6cc10b9dc0ea2ac64898df3287.md) |<span style='color: green'>(-31 [-1.4%])</span> 2,186 |  2,579,903 | <span style='color: green'>(-3 [-0.6%])</span> 476 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/11e3769e6405ac6cc10b9dc0ea2ac64898df3287

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29597046094)
