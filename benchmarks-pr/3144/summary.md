| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3144/fibonacci-3741601f1c5183c0211a6926d62fb52343c6da9d.md) |<span style='color: green'>(-2 [-0.1%])</span> 1,672 |  12,000,265 | <span style='color: green'>(-4 [-1.1%])</span> 365 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3144/keccak-3741601f1c5183c0211a6926d62fb52343c6da9d.md) |<span style='color: red'>(+35 [+0.4%])</span> 9,630 |  18,655,329 | <span style='color: red'>(+10 [+0.7%])</span> 1,544 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3144/sha2_bench-3741601f1c5183c0211a6926d62fb52343c6da9d.md) |<span style='color: red'>(+76 [+1.4%])</span> 5,332 |  14,793,960 | <span style='color: red'>(+3 [+0.5%])</span> 593 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3144/regex-3741601f1c5183c0211a6926d62fb52343c6da9d.md) |<span style='color: green'>(-5 [-0.7%])</span> 689 |  4,137,067 | <span style='color: red'>(+3 [+1.4%])</span> 221 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3144/ecrecover-3741601f1c5183c0211a6926d62fb52343c6da9d.md) |<span style='color: green'>(-1 [-0.2%])</span> 439 |  123,583 | <span style='color: green'>(-5 [-2.6%])</span> 190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3144/pairing-3741601f1c5183c0211a6926d62fb52343c6da9d.md) |<span style='color: green'>(-9 [-1.5%])</span> 578 |  1,745,757 |  194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3144/kitchen_sink-3741601f1c5183c0211a6926d62fb52343c6da9d.md) |<span style='color: green'>(-13 [-0.6%])</span> 2,288 |  2,579,903 | <span style='color: green'>(-4 [-0.8%])</span> 496 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3741601f1c5183c0211a6926d62fb52343c6da9d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33542960646)
