| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3157/fibonacci-b4e97a7aeca0482b0841af64f402c43431300338.md) |<span style='color: green'>(-23 [-1.4%])</span> 1,652 |  12,000,265 | <span style='color: green'>(-4 [-1.1%])</span> 368 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3157/keccak-b4e97a7aeca0482b0841af64f402c43431300338.md) |<span style='color: red'>(+67 [+0.7%])</span> 9,545 |  18,655,329 | <span style='color: green'>(-7 [-0.5%])</span> 1,533 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3157/sha2_bench-b4e97a7aeca0482b0841af64f402c43431300338.md) |<span style='color: red'>(+102 [+1.9%])</span> 5,345 |  14,793,960 | <span style='color: red'>(+3 [+0.5%])</span> 591 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3157/regex-b4e97a7aeca0482b0841af64f402c43431300338.md) |<span style='color: red'>(+8 [+1.2%])</span> 700 |  4,137,067 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3157/ecrecover-b4e97a7aeca0482b0841af64f402c43431300338.md) |<span style='color: red'>(+1 [+0.2%])</span> 429 |  123,583 | <span style='color: red'>(+5 [+2.7%])</span> 192 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3157/pairing-b4e97a7aeca0482b0841af64f402c43431300338.md) |<span style='color: green'>(-13 [-2.2%])</span> 576 |  1,745,757 | <span style='color: green'>(-2 [-1.0%])</span> 195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3157/kitchen_sink-b4e97a7aeca0482b0841af64f402c43431300338.md) |<span style='color: red'>(+13 [+0.6%])</span> 2,301 |  2,579,903 | <span style='color: green'>(-4 [-0.8%])</span> 493 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b4e97a7aeca0482b0841af64f402c43431300338

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/34984375986)
