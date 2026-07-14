| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3016/fibonacci-4fcb1e5994d1301d28f25d75e3c8a0b9c8d2cc17.md) |<span style='color: red'>(+46 [+1.5%])</span> 3,080 |  12,000,265 | <span style='color: red'>(+2 [+0.3%])</span> 673 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3016/keccak-4fcb1e5994d1301d28f25d75e3c8a0b9c8d2cc17.md) |<span style='color: red'>(+189 [+1.2%])</span> 16,518 |  18,655,329 | <span style='color: red'>(+39 [+1.3%])</span> 3,067 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3016/sha2_bench-4fcb1e5994d1301d28f25d75e3c8a0b9c8d2cc17.md) |<span style='color: red'>(+264 [+2.9%])</span> 9,401 |  14,793,960 | <span style='color: red'>(+21 [+1.9%])</span> 1,144 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3016/regex-4fcb1e5994d1301d28f25d75e3c8a0b9c8d2cc17.md) |<span style='color: red'>(+9 [+0.8%])</span> 1,176 |  4,137,067 | <span style='color: red'>(+10 [+2.8%])</span> 361 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3016/ecrecover-4fcb1e5994d1301d28f25d75e3c8a0b9c8d2cc17.md) |<span style='color: red'>(+3 [+0.5%])</span> 601 |  123,583 | <span style='color: green'>(-2 [-0.7%])</span> 282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3016/pairing-4fcb1e5994d1301d28f25d75e3c8a0b9c8d2cc17.md) |<span style='color: red'>(+16 [+1.7%])</span> 947 |  1,745,757 | <span style='color: red'>(+5 [+1.6%])</span> 313 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3016/kitchen_sink-4fcb1e5994d1301d28f25d75e3c8a0b9c8d2cc17.md) |<span style='color: green'>(-6 [-0.1%])</span> 4,119 |  2,579,903 | <span style='color: red'>(+9 [+1.0%])</span> 889 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4fcb1e5994d1301d28f25d75e3c8a0b9c8d2cc17

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29363464085)
