| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/fibonacci-2ffd4d56d79a2c01a43235bcdbbb13e5bf28adfd.md) | 1,587 |  12,000,265 | <span style='color: red'>(+3 [+0.8%])</span> 364 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/keccak-2ffd4d56d79a2c01a43235bcdbbb13e5bf28adfd.md) |<span style='color: red'>(+107 [+1.2%])</span> 9,362 |  18,655,329 | <span style='color: red'>(+18 [+1.2%])</span> 1,533 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/sha2_bench-2ffd4d56d79a2c01a43235bcdbbb13e5bf28adfd.md) |<span style='color: red'>(+63 [+1.3%])</span> 4,938 |  14,793,960 | <span style='color: red'>(+6 [+1.0%])</span> 578 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/regex-2ffd4d56d79a2c01a43235bcdbbb13e5bf28adfd.md) |<span style='color: red'>(+1 [+0.2%])</span> 663 |  4,137,067 |  210 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/ecrecover-2ffd4d56d79a2c01a43235bcdbbb13e5bf28adfd.md) |<span style='color: red'>(+7 [+1.6%])</span> 434 |  123,583 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/pairing-2ffd4d56d79a2c01a43235bcdbbb13e5bf28adfd.md) |<span style='color: red'>(+7 [+1.2%])</span> 577 |  1,745,757 | <span style='color: green'>(-2 [-1.0%])</span> 190 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/kitchen_sink-2ffd4d56d79a2c01a43235bcdbbb13e5bf28adfd.md) | 2,213 |  2,579,903 | <span style='color: red'>(+3 [+0.6%])</span> 480 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2ffd4d56d79a2c01a43235bcdbbb13e5bf28adfd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30108898435)
