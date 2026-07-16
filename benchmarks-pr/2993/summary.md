| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/fibonacci-b0fbba75907517d1ca4f8bb00339c9e881bdf014.md) | 408 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/keccak-b0fbba75907517d1ca4f8bb00339c9e881bdf014.md) | 8,726 |  14,365,133 |  1,533 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/sha2_bench-b0fbba75907517d1ca4f8bb00339c9e881bdf014.md) | 4,238 |  11,167,961 |  527 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/regex-b0fbba75907517d1ca4f8bb00339c9e881bdf014.md) | 579 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/ecrecover-b0fbba75907517d1ca4f8bb00339c9e881bdf014.md) | 221 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/pairing-b0fbba75907517d1ca4f8bb00339c9e881bdf014.md) | 294 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/kitchen_sink-b0fbba75907517d1ca4f8bb00339c9e881bdf014.md) | 1,909 |  1,979,971 |  456 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b0fbba75907517d1ca4f8bb00339c9e881bdf014

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29515418994)
