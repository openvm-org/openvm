| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3086/fibonacci-560d7a96dfb427816fdce755872f7905c328c745.md) |<span style='color: green'>(-4 [-0.3%])</span> 1,579 |  12,000,265 | <span style='color: green'>(-1 [-0.3%])</span> 361 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3086/keccak-560d7a96dfb427816fdce755872f7905c328c745.md) |<span style='color: green'>(-116 [-1.2%])</span> 9,262 |  18,655,329 | <span style='color: green'>(-39 [-2.5%])</span> 1,512 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3086/sha2_bench-560d7a96dfb427816fdce755872f7905c328c745.md) |<span style='color: red'>(+14 [+0.3%])</span> 4,973 |  14,793,960 | <span style='color: red'>(+2 [+0.3%])</span> 580 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3086/regex-560d7a96dfb427816fdce755872f7905c328c745.md) |<span style='color: green'>(-2 [-0.3%])</span> 662 |  4,137,067 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3086/ecrecover-560d7a96dfb427816fdce755872f7905c328c745.md) |<span style='color: red'>(+9 [+2.1%])</span> 443 |  123,583 | <span style='color: green'>(-1 [-0.5%])</span> 183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3086/pairing-560d7a96dfb427816fdce755872f7905c328c745.md) |<span style='color: red'>(+34 [+6.1%])</span> 590 |  1,745,757 | <span style='color: green'>(-1 [-0.5%])</span> 187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3086/kitchen_sink-560d7a96dfb427816fdce755872f7905c328c745.md) |<span style='color: green'>(-15 [-0.7%])</span> 2,200 |  2,579,903 | <span style='color: red'>(+3 [+0.6%])</span> 476 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/560d7a96dfb427816fdce755872f7905c328c745

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30575277465)
