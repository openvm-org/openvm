| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/fibonacci-b280f9cc59602c7ee6e5a26b3f4dd4200b66d462.md) | 1,045 |  4,000,051 |  396 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/keccak-b280f9cc59602c7ee6e5a26b3f4dd4200b66d462.md) | 16,956 |  14,365,133 |  3,132 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/sha2_bench-b280f9cc59602c7ee6e5a26b3f4dd4200b66d462.md) | 8,230 |  11,167,961 |  1,005 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/regex-b280f9cc59602c7ee6e5a26b3f4dd4200b66d462.md) | 1,233 |  4,090,656 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/ecrecover-b280f9cc59602c7ee6e5a26b3f4dd4200b66d462.md) | 438 |  112,210 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/pairing-b280f9cc59602c7ee6e5a26b3f4dd4200b66d462.md) | 598 |  592,827 |  303 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/kitchen_sink-b280f9cc59602c7ee6e5a26b3f4dd4200b66d462.md) | 3,870 |  1,979,971 |  858 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b280f9cc59602c7ee6e5a26b3f4dd4200b66d462

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28062798849)
