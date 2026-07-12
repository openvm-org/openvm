| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/fibonacci-0a8a34cdc6970b264d6803066693ea8ae395381c.md) |<span style='color: red'>(+80 [+2.6%])</span> 3,114 |  12,000,265 | <span style='color: red'>(+8 [+1.2%])</span> 679 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/keccak-0a8a34cdc6970b264d6803066693ea8ae395381c.md) |<span style='color: green'>(-31 [-0.2%])</span> 16,298 |  18,655,329 |  3,027 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/sha2_bench-0a8a34cdc6970b264d6803066693ea8ae395381c.md) |<span style='color: red'>(+112 [+1.2%])</span> 9,249 |  14,793,960 | <span style='color: red'>(+6 [+0.5%])</span> 1,129 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/regex-0a8a34cdc6970b264d6803066693ea8ae395381c.md) |<span style='color: red'>(+33 [+2.8%])</span> 1,200 |  4,137,067 | <span style='color: red'>(+1 [+0.3%])</span> 352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/ecrecover-0a8a34cdc6970b264d6803066693ea8ae395381c.md) |<span style='color: green'>(-32 [-5.4%])</span> 566 |  123,583 | <span style='color: green'>(-2 [-0.7%])</span> 282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/pairing-0a8a34cdc6970b264d6803066693ea8ae395381c.md) | 931 |  1,745,757 | <span style='color: red'>(+3 [+1.0%])</span> 311 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/kitchen_sink-0a8a34cdc6970b264d6803066693ea8ae395381c.md) |<span style='color: red'>(+382 [+9.3%])</span> 4,507 |  2,579,903 | <span style='color: red'>(+5 [+0.6%])</span> 885 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0a8a34cdc6970b264d6803066693ea8ae395381c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29211823945)
