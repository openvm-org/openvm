| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3000/fibonacci-607d0ae0a982abe812d74ac6747cacda6a2aaff4.md) |<span style='color: red'>(+26 [+0.9%])</span> 3,060 |  12,000,265 | <span style='color: green'>(-1 [-0.1%])</span> 670 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3000/keccak-607d0ae0a982abe812d74ac6747cacda6a2aaff4.md) |<span style='color: red'>(+32 [+0.2%])</span> 16,361 |  18,655,329 |  3,031 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3000/sha2_bench-607d0ae0a982abe812d74ac6747cacda6a2aaff4.md) |<span style='color: red'>(+62 [+0.7%])</span> 9,199 |  14,793,960 | <span style='color: red'>(+2 [+0.2%])</span> 1,125 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3000/regex-607d0ae0a982abe812d74ac6747cacda6a2aaff4.md) |<span style='color: green'>(-3 [-0.3%])</span> 1,164 |  4,137,067 | <span style='color: red'>(+7 [+2.0%])</span> 358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3000/ecrecover-607d0ae0a982abe812d74ac6747cacda6a2aaff4.md) |<span style='color: red'>(+4 [+0.7%])</span> 602 |  123,583 | <span style='color: red'>(+1 [+0.4%])</span> 285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3000/pairing-607d0ae0a982abe812d74ac6747cacda6a2aaff4.md) |<span style='color: red'>(+10 [+1.1%])</span> 941 |  1,745,757 |  308 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3000/kitchen_sink-607d0ae0a982abe812d74ac6747cacda6a2aaff4.md) |<span style='color: green'>(-36 [-0.9%])</span> 4,089 |  2,579,903 |  880 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/607d0ae0a982abe812d74ac6747cacda6a2aaff4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29072036398)
