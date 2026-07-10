| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/fibonacci-b86f0c1977ebea47a90ca83e7bfd83ea267c9419.md) | 851 |  4,000,051 |  387 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/keccak-b86f0c1977ebea47a90ca83e7bfd83ea267c9419.md) | 15,766 |  14,365,133 |  3,042 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/sha2_bench-b86f0c1977ebea47a90ca83e7bfd83ea267c9419.md) | 7,842 |  11,167,961 |  1,007 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/regex-b86f0c1977ebea47a90ca83e7bfd83ea267c9419.md) | 1,038 |  4,090,656 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/ecrecover-b86f0c1977ebea47a90ca83e7bfd83ea267c9419.md) | 310 |  112,210 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/pairing-b86f0c1977ebea47a90ca83e7bfd83ea267c9419.md) | 439 |  592,827 |  300 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/kitchen_sink-b86f0c1977ebea47a90ca83e7bfd83ea267c9419.md) | 3,745 |  1,979,971 |  863 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b86f0c1977ebea47a90ca83e7bfd83ea267c9419

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29122807082)
