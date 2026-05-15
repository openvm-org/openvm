| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/fibonacci-9c6316ae8343eadf8e2b35633beabeb53ae07490.md) | 1,830 |  4,000,051 |  435 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/keccak-9c6316ae8343eadf8e2b35633beabeb53ae07490.md) | 13,928 |  14,365,133 |  2,364 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/sha2_bench-9c6316ae8343eadf8e2b35633beabeb53ae07490.md) | 8,310 |  11,167,961 |  921 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/regex-9c6316ae8343eadf8e2b35633beabeb53ae07490.md) | 1,566 |  4,090,656 |  363 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/ecrecover-9c6316ae8343eadf8e2b35633beabeb53ae07490.md) | 602 |  112,210 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/pairing-9c6316ae8343eadf8e2b35633beabeb53ae07490.md) | 732 |  592,827 |  260 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/kitchen_sink-9c6316ae8343eadf8e2b35633beabeb53ae07490.md) | 1,896 |  1,979,971 |  414 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9c6316ae8343eadf8e2b35633beabeb53ae07490

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25935634057)
