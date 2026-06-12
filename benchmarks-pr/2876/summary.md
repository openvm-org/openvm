| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/fibonacci-565bc3aa21c8891d1798411ceb582dbf57d28b3e.md) | 1,682 |  4,000,051 |  533 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/keccak-565bc3aa21c8891d1798411ceb582dbf57d28b3e.md) | 16,478 |  14,365,133 |  3,060 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/sha2_bench-565bc3aa21c8891d1798411ceb582dbf57d28b3e.md) | 10,420 |  11,167,961 |  1,937 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/regex-565bc3aa21c8891d1798411ceb582dbf57d28b3e.md) | 1,538 |  4,090,656 |  435 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/ecrecover-565bc3aa21c8891d1798411ceb582dbf57d28b3e.md) | 482 |  112,210 |  311 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/pairing-565bc3aa21c8891d1798411ceb582dbf57d28b3e.md) | 617 |  592,827 |  292 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/kitchen_sink-565bc3aa21c8891d1798411ceb582dbf57d28b3e.md) | 3,923 |  1,979,971 |  856 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/565bc3aa21c8891d1798411ceb582dbf57d28b3e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27432526790)
