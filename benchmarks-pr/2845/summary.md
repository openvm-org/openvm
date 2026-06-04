| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/fibonacci-7f75fa53866dab6b275b4015131982b4d1c13f27.md) | 3,726 |  12,000,265 | <span style='color: green'>(-3575 [-79.7%])</span> 911 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/keccak-7f75fa53866dab6b275b4015131982b4d1c13f27.md) | 18,762 |  18,655,329 |  3,318 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/sha2_bench-7f75fa53866dab6b275b4015131982b4d1c13f27.md) | 10,319 |  14,793,960 |  1,472 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/regex-7f75fa53866dab6b275b4015131982b4d1c13f27.md) | 1,410 |  4,137,067 | <span style='color: green'>(-11640 [-97.0%])</span> 357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/ecrecover-7f75fa53866dab6b275b4015131982b4d1c13f27.md) | 601 |  123,583 | <span style='color: green'>(-5602 [-95.7%])</span> 254 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/pairing-7f75fa53866dab6b275b4015131982b4d1c13f27.md) | 900 |  1,745,757 | <span style='color: green'>(-6115 [-95.8%])</span> 265 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2845/kitchen_sink-7f75fa53866dab6b275b4015131982b4d1c13f27.md) | 1,910 |  2,579,903 |  415 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7f75fa53866dab6b275b4015131982b4d1c13f27

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26984933336)
