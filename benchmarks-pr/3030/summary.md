| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/fibonacci-64cc734946fa93cd50ba730d992f037f8fb25b1c.md) |<span style='color: red'>(+29 [+1.8%])</span> 1,603 |  12,000,265 | <span style='color: red'>(+4 [+1.1%])</span> 367 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/keccak-64cc734946fa93cd50ba730d992f037f8fb25b1c.md) |<span style='color: red'>(+105 [+1.1%])</span> 9,379 |  18,655,329 | <span style='color: red'>(+20 [+1.3%])</span> 1,534 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/sha2_bench-64cc734946fa93cd50ba730d992f037f8fb25b1c.md) |<span style='color: green'>(-23 [-0.5%])</span> 4,924 |  14,793,960 | <span style='color: red'>(+8 [+1.4%])</span> 584 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/regex-64cc734946fa93cd50ba730d992f037f8fb25b1c.md) |<span style='color: green'>(-9 [-1.4%])</span> 652 |  4,137,067 | <span style='color: green'>(-1 [-0.5%])</span> 211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/ecrecover-64cc734946fa93cd50ba730d992f037f8fb25b1c.md) |<span style='color: red'>(+5 [+1.2%])</span> 436 |  123,583 | <span style='color: red'>(+2 [+1.1%])</span> 184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/pairing-64cc734946fa93cd50ba730d992f037f8fb25b1c.md) |<span style='color: red'>(+5 [+0.9%])</span> 574 |  1,745,757 | <span style='color: red'>(+3 [+1.6%])</span> 193 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3030/kitchen_sink-64cc734946fa93cd50ba730d992f037f8fb25b1c.md) |<span style='color: green'>(-14 [-0.6%])</span> 2,189 |  2,579,903 | <span style='color: green'>(-2 [-0.4%])</span> 473 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/64cc734946fa93cd50ba730d992f037f8fb25b1c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30390582128)
