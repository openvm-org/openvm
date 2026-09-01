| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/fibonacci-78cf9069b53f78f743863e08e4e782720dd82bc1.md) |<span style='color: red'>(+9 [+0.5%])</span> 1,676 |  12,000,265 | <span style='color: red'>(+7 [+1.9%])</span> 375 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/keccak-78cf9069b53f78f743863e08e4e782720dd82bc1.md) |<span style='color: green'>(-80 [-0.8%])</span> 9,559 |  18,655,329 | <span style='color: red'>(+8 [+0.5%])</span> 1,559 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/sha2_bench-78cf9069b53f78f743863e08e4e782720dd82bc1.md) |<span style='color: red'>(+51 [+1.0%])</span> 5,327 |  14,793,960 | <span style='color: green'>(-1 [-0.2%])</span> 595 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/regex-78cf9069b53f78f743863e08e4e782720dd82bc1.md) |<span style='color: red'>(+24 [+3.5%])</span> 710 |  4,137,067 | <span style='color: red'>(+4 [+1.9%])</span> 219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/ecrecover-78cf9069b53f78f743863e08e4e782720dd82bc1.md) |<span style='color: green'>(-18 [-4.0%])</span> 429 |  123,583 | <span style='color: green'>(-5 [-2.6%])</span> 187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/pairing-78cf9069b53f78f743863e08e4e782720dd82bc1.md) |<span style='color: red'>(+1 [+0.2%])</span> 584 |  1,745,757 | <span style='color: green'>(-3 [-1.5%])</span> 195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/kitchen_sink-78cf9069b53f78f743863e08e4e782720dd82bc1.md) |<span style='color: red'>(+11 [+0.5%])</span> 2,314 |  2,579,903 | <span style='color: red'>(+2 [+0.4%])</span> 498 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/78cf9069b53f78f743863e08e4e782720dd82bc1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33522710021)
