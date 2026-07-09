| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2990/fibonacci-1647b664a724f20734d97ca5023a798e20922576.md) |<span style='color: green'>(-15 [-0.5%])</span> 3,088 |  12,000,265 | <span style='color: green'>(-8 [-1.2%])</span> 680 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2990/keccak-1647b664a724f20734d97ca5023a798e20922576.md) |<span style='color: red'>(+65 [+0.4%])</span> 16,490 |  18,655,329 | <span style='color: red'>(+18 [+0.6%])</span> 3,064 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2990/sha2_bench-1647b664a724f20734d97ca5023a798e20922576.md) |<span style='color: red'>(+233 [+2.6%])</span> 9,338 |  14,793,960 | <span style='color: green'>(-6 [-0.5%])</span> 1,136 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2990/regex-1647b664a724f20734d97ca5023a798e20922576.md) |<span style='color: red'>(+95 [+8.2%])</span> 1,247 |  4,137,067 | <span style='color: red'>(+6 [+1.7%])</span> 354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2990/ecrecover-1647b664a724f20734d97ca5023a798e20922576.md) |<span style='color: red'>(+20 [+3.4%])</span> 614 |  123,583 | <span style='color: red'>(+11 [+4.0%])</span> 289 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2990/pairing-1647b664a724f20734d97ca5023a798e20922576.md) |<span style='color: green'>(-1 [-0.1%])</span> 954 |  1,745,757 | <span style='color: green'>(-8 [-2.6%])</span> 305 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2990/kitchen_sink-1647b664a724f20734d97ca5023a798e20922576.md) |<span style='color: red'>(+407 [+9.7%])</span> 4,595 |  2,579,903 | <span style='color: green'>(-11 [-1.2%])</span> 889 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1647b664a724f20734d97ca5023a798e20922576

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29051418705)
