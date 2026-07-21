| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/fibonacci-88c7ea881fc6b6daeb0155d940c4a4030a569579.md) |<span style='color: green'>(-15 [-0.9%])</span> 1,571 |  12,000,265 | <span style='color: green'>(-4 [-1.1%])</span> 357 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/keccak-88c7ea881fc6b6daeb0155d940c4a4030a569579.md) |<span style='color: green'>(-81 [-0.9%])</span> 9,174 |  18,655,329 | <span style='color: green'>(-6 [-0.4%])</span> 1,509 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/sha2_bench-88c7ea881fc6b6daeb0155d940c4a4030a569579.md) |<span style='color: red'>(+62 [+1.3%])</span> 4,937 |  14,793,960 | <span style='color: red'>(+2 [+0.3%])</span> 574 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/regex-88c7ea881fc6b6daeb0155d940c4a4030a569579.md) |<span style='color: red'>(+6 [+0.9%])</span> 668 |  4,137,067 | <span style='color: red'>(+1 [+0.5%])</span> 211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/ecrecover-88c7ea881fc6b6daeb0155d940c4a4030a569579.md) |<span style='color: red'>(+9 [+2.1%])</span> 436 |  123,583 | <span style='color: red'>(+1 [+0.5%])</span> 186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/pairing-88c7ea881fc6b6daeb0155d940c4a4030a569579.md) |<span style='color: green'>(-14 [-2.5%])</span> 556 |  1,745,757 | <span style='color: green'>(-2 [-1.0%])</span> 190 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/kitchen_sink-88c7ea881fc6b6daeb0155d940c4a4030a569579.md) | 2,216 |  2,579,903 | <span style='color: red'>(+3 [+0.6%])</span> 480 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/88c7ea881fc6b6daeb0155d940c4a4030a569579

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29855320132)
