| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci-d6e8a8d3600be21da1af37598f2ca06930c7186b.md) | 1,904 |  4,000,051 |  533 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/keccak-d6e8a8d3600be21da1af37598f2ca06930c7186b.md) | 13,713 |  14,365,133 |  2,272 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/sha2_bench-d6e8a8d3600be21da1af37598f2ca06930c7186b.md) | 9,550 |  11,167,961 |  1,430 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex-d6e8a8d3600be21da1af37598f2ca06930c7186b.md) | 1,632 |  4,090,656 |  386 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover-d6e8a8d3600be21da1af37598f2ca06930c7186b.md) | 646 |  112,210 |  292 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing-d6e8a8d3600be21da1af37598f2ca06930c7186b.md) | 752 |  592,827 |  276 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink-d6e8a8d3600be21da1af37598f2ca06930c7186b.md) | 2,041 |  1,979,971 |  429 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d6e8a8d3600be21da1af37598f2ca06930c7186b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25752742744)
