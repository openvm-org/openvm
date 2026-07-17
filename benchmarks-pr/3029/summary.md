| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/fibonacci-312a3f624a92cf9713537ef55c1608d7827f0718.md) |<span style='color: green'>(-25 [-1.6%])</span> 1,579 |  12,000,265 | <span style='color: green'>(-5 [-1.4%])</span> 359 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/keccak-312a3f624a92cf9713537ef55c1608d7827f0718.md) |<span style='color: green'>(-98 [-1.0%])</span> 9,306 |  18,655,329 | <span style='color: green'>(-15 [-1.0%])</span> 1,531 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/sha2_bench-312a3f624a92cf9713537ef55c1608d7827f0718.md) |<span style='color: red'>(+26 [+0.5%])</span> 4,880 |  14,793,960 | <span style='color: green'>(-2 [-0.3%])</span> 572 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/regex-312a3f624a92cf9713537ef55c1608d7827f0718.md) |<span style='color: red'>(+13 [+2.0%])</span> 666 |  4,137,067 | <span style='color: red'>(+1 [+0.5%])</span> 214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/ecrecover-312a3f624a92cf9713537ef55c1608d7827f0718.md) |<span style='color: green'>(-10 [-2.3%])</span> 425 |  123,583 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/pairing-312a3f624a92cf9713537ef55c1608d7827f0718.md) |<span style='color: green'>(-39 [-6.5%])</span> 559 |  1,745,757 | <span style='color: green'>(-4 [-2.1%])</span> 187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/kitchen_sink-312a3f624a92cf9713537ef55c1608d7827f0718.md) |<span style='color: green'>(-17 [-0.8%])</span> 2,200 |  2,579,903 | <span style='color: green'>(-4 [-0.8%])</span> 475 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/312a3f624a92cf9713537ef55c1608d7827f0718

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29593700543)
