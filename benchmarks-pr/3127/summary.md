| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3127/fibonacci-80d86c62da577b962ce518f153acc1339e6435e1.md) |<span style='color: red'>(+104 [+6.6%])</span> 1,674 |  12,000,265 | <span style='color: red'>(+11 [+3.1%])</span> 370 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3127/keccak-80d86c62da577b962ce518f153acc1339e6435e1.md) |<span style='color: red'>(+309 [+3.3%])</span> 9,631 |  18,655,329 | <span style='color: red'>(+3 [+0.2%])</span> 1,539 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3127/sha2_bench-80d86c62da577b962ce518f153acc1339e6435e1.md) |<span style='color: red'>(+487 [+10.0%])</span> 5,378 |  14,793,960 | <span style='color: red'>(+8 [+1.4%])</span> 587 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3127/regex-80d86c62da577b962ce518f153acc1339e6435e1.md) |<span style='color: red'>(+41 [+6.2%])</span> 697 |  4,137,067 | <span style='color: red'>(+4 [+1.9%])</span> 217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3127/ecrecover-80d86c62da577b962ce518f153acc1339e6435e1.md) | 435 |  123,583 | <span style='color: red'>(+7 [+3.8%])</span> 192 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3127/pairing-80d86c62da577b962ce518f153acc1339e6435e1.md) |<span style='color: red'>(+29 [+5.2%])</span> 590 |  1,745,757 | <span style='color: red'>(+1 [+0.5%])</span> 195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3127/kitchen_sink-80d86c62da577b962ce518f153acc1339e6435e1.md) |<span style='color: red'>(+130 [+5.9%])</span> 2,323 |  2,579,903 | <span style='color: red'>(+26 [+5.5%])</span> 498 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/80d86c62da577b962ce518f153acc1339e6435e1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32525250152)
