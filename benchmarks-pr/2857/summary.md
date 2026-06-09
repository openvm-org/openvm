| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2857/fibonacci-907da84aed491fff7dc2cfbba3d5de00b5d3f79d.md) | 3,776 |  12,000,265 |  937 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2857/keccak-907da84aed491fff7dc2cfbba3d5de00b5d3f79d.md) | 18,332 |  18,655,329 |  3,324 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2857/sha2_bench-907da84aed491fff7dc2cfbba3d5de00b5d3f79d.md) | 9,910 |  14,793,960 |  1,446 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2857/regex-907da84aed491fff7dc2cfbba3d5de00b5d3f79d.md) | 1,406 |  4,137,067 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2857/ecrecover-907da84aed491fff7dc2cfbba3d5de00b5d3f79d.md) | 599 |  123,583 |  244 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2857/pairing-907da84aed491fff7dc2cfbba3d5de00b5d3f79d.md) | 881 |  1,745,757 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2857/kitchen_sink-907da84aed491fff7dc2cfbba3d5de00b5d3f79d.md) | 3,907 |  2,579,903 |  972 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/907da84aed491fff7dc2cfbba3d5de00b5d3f79d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27229863744)
