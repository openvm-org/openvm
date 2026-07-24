| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/fibonacci-2ffd4d56d79a2c01a43235bcdbbb13e5bf28adfd.md) |<span style='color: green'>(-9 [-0.6%])</span> 1,577 |  12,000,265 | <span style='color: green'>(-4 [-1.1%])</span> 357 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/keccak-2ffd4d56d79a2c01a43235bcdbbb13e5bf28adfd.md) | 9,259 |  18,655,329 | <span style='color: green'>(-4 [-0.3%])</span> 1,511 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/sha2_bench-2ffd4d56d79a2c01a43235bcdbbb13e5bf28adfd.md) |<span style='color: red'>(+52 [+1.1%])</span> 4,927 |  14,793,960 | <span style='color: red'>(+7 [+1.2%])</span> 579 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/regex-2ffd4d56d79a2c01a43235bcdbbb13e5bf28adfd.md) |<span style='color: green'>(-5 [-0.8%])</span> 657 |  4,137,067 | <span style='color: red'>(+6 [+2.9%])</span> 216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/ecrecover-2ffd4d56d79a2c01a43235bcdbbb13e5bf28adfd.md) |<span style='color: red'>(+7 [+1.6%])</span> 434 |  123,583 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/pairing-2ffd4d56d79a2c01a43235bcdbbb13e5bf28adfd.md) |<span style='color: red'>(+6 [+1.1%])</span> 576 |  1,745,757 | <span style='color: green'>(-3 [-1.6%])</span> 189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/kitchen_sink-2ffd4d56d79a2c01a43235bcdbbb13e5bf28adfd.md) |<span style='color: red'>(+5 [+0.2%])</span> 2,219 |  2,579,903 | <span style='color: red'>(+4 [+0.8%])</span> 481 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2ffd4d56d79a2c01a43235bcdbbb13e5bf28adfd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30103436935)
