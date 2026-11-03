| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3189/fibonacci-5ad1a0027dd6aa063b4d6f9201f95fdbf8ddce1a.md) |<span style='color: green'>(-20 [-1.2%])</span> 1,677 |  12,000,265 | <span style='color: green'>(-9 [-2.4%])</span> 368 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3189/keccak-5ad1a0027dd6aa063b4d6f9201f95fdbf8ddce1a.md) |<span style='color: red'>(+96 [+1.0%])</span> 9,678 |  18,655,329 | <span style='color: red'>(+10 [+0.6%])</span> 1,557 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3189/sha2_bench-5ad1a0027dd6aa063b4d6f9201f95fdbf8ddce1a.md) |<span style='color: red'>(+68 [+1.3%])</span> 5,311 |  14,793,960 | <span style='color: red'>(+7 [+1.2%])</span> 593 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3189/regex-5ad1a0027dd6aa063b4d6f9201f95fdbf8ddce1a.md) |<span style='color: red'>(+8 [+1.2%])</span> 696 |  4,137,067 | <span style='color: red'>(+6 [+2.8%])</span> 220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3189/ecrecover-5ad1a0027dd6aa063b4d6f9201f95fdbf8ddce1a.md) |<span style='color: green'>(-2 [-0.5%])</span> 441 |  123,583 | <span style='color: green'>(-3 [-1.6%])</span> 189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3189/pairing-5ad1a0027dd6aa063b4d6f9201f95fdbf8ddce1a.md) |<span style='color: red'>(+8 [+1.4%])</span> 580 |  1,745,757 | <span style='color: red'>(+3 [+1.6%])</span> 195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3189/kitchen_sink-5ad1a0027dd6aa063b4d6f9201f95fdbf8ddce1a.md) |<span style='color: green'>(-8 [-0.3%])</span> 2,299 |  2,579,903 | <span style='color: green'>(-2 [-0.4%])</span> 493 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5ad1a0027dd6aa063b4d6f9201f95fdbf8ddce1a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/41676856183)
