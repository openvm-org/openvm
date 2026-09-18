| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3159/fibonacci-2a94f29b8c2223bcdc70abea624e6cef59b603c2.md) |<span style='color: green'>(-19 [-1.1%])</span> 1,669 |  12,000,265 | <span style='color: green'>(-3 [-0.8%])</span> 369 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3159/keccak-2a94f29b8c2223bcdc70abea624e6cef59b603c2.md) |<span style='color: green'>(-190 [-1.9%])</span> 9,563 |  18,655,329 | <span style='color: green'>(-20 [-1.3%])</span> 1,536 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3159/sha2_bench-2a94f29b8c2223bcdc70abea624e6cef59b603c2.md) |<span style='color: red'>(+13 [+0.2%])</span> 5,259 |  14,793,960 | <span style='color: green'>(-2 [-0.3%])</span> 588 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3159/regex-2a94f29b8c2223bcdc70abea624e6cef59b603c2.md) |<span style='color: green'>(-16 [-2.3%])</span> 683 |  4,137,067 | <span style='color: red'>(+6 [+2.8%])</span> 221 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3159/ecrecover-2a94f29b8c2223bcdc70abea624e6cef59b603c2.md) |<span style='color: green'>(-5 [-1.1%])</span> 439 |  123,583 | <span style='color: green'>(-4 [-2.1%])</span> 189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3159/pairing-2a94f29b8c2223bcdc70abea624e6cef59b603c2.md) |<span style='color: green'>(-17 [-2.9%])</span> 563 |  1,745,757 |  196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3159/kitchen_sink-2a94f29b8c2223bcdc70abea624e6cef59b603c2.md) |<span style='color: red'>(+32 [+1.4%])</span> 2,315 |  2,579,903 | <span style='color: red'>(+9 [+1.8%])</span> 499 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2a94f29b8c2223bcdc70abea624e6cef59b603c2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/35358076648)
