| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-800a67516ce719879df732ffba67c11d25318bd5.md) | 464 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-800a67516ce719879df732ffba67c11d25318bd5.md) | 7,402 |  14,365,133 |  1,544 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-800a67516ce719879df732ffba67c11d25318bd5.md) | 4,126 |  11,167,961 |  516 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-800a67516ce719879df732ffba67c11d25318bd5.md) | 664 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-800a67516ce719879df732ffba67c11d25318bd5.md) | 203 |  112,210 |  198 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-800a67516ce719879df732ffba67c11d25318bd5.md) | 238 |  592,827 |  199 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-800a67516ce719879df732ffba67c11d25318bd5.md) | 2,038 |  1,979,971 |  529 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/800a67516ce719879df732ffba67c11d25318bd5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31676189641)
