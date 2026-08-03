| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3089/fibonacci-0685697886c49aac7652d91e139ff1ade11bcaf2.md) |<span style='color: green'>(-3 [-0.2%])</span> 1,577 |  12,000,265 | <span style='color: green'>(-5 [-1.4%])</span> 359 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3089/keccak-0685697886c49aac7652d91e139ff1ade11bcaf2.md) |<span style='color: green'>(-34 [-0.4%])</span> 9,280 |  18,655,329 | <span style='color: green'>(-12 [-0.8%])</span> 1,511 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3089/sha2_bench-0685697886c49aac7652d91e139ff1ade11bcaf2.md) |<span style='color: red'>(+50 [+1.0%])</span> 4,965 |  14,793,960 | <span style='color: red'>(+3 [+0.5%])</span> 580 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3089/regex-0685697886c49aac7652d91e139ff1ade11bcaf2.md) |<span style='color: red'>(+1 [+0.2%])</span> 662 |  4,137,067 |  211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3089/ecrecover-0685697886c49aac7652d91e139ff1ade11bcaf2.md) |<span style='color: red'>(+4 [+0.9%])</span> 436 |  123,583 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3089/pairing-0685697886c49aac7652d91e139ff1ade11bcaf2.md) |<span style='color: red'>(+31 [+5.5%])</span> 591 |  1,745,757 | <span style='color: red'>(+1 [+0.5%])</span> 192 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3089/kitchen_sink-0685697886c49aac7652d91e139ff1ade11bcaf2.md) | 2,218 |  2,579,903 | <span style='color: red'>(+5 [+1.1%])</span> 479 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0685697886c49aac7652d91e139ff1ade11bcaf2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30802663384)
