| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/fibonacci-512c3448ad7ab7b257cd1b2755d8c1cbcdb37b31.md) | 866 |  4,000,051 |  389 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/keccak-512c3448ad7ab7b257cd1b2755d8c1cbcdb37b31.md) | 15,549 |  14,365,133 |  3,009 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/sha2_bench-512c3448ad7ab7b257cd1b2755d8c1cbcdb37b31.md) | 7,870 |  11,167,961 |  999 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/regex-512c3448ad7ab7b257cd1b2755d8c1cbcdb37b31.md) | 1,015 |  4,090,656 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/ecrecover-512c3448ad7ab7b257cd1b2755d8c1cbcdb37b31.md) | 306 |  112,210 |  288 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/pairing-512c3448ad7ab7b257cd1b2755d8c1cbcdb37b31.md) | 435 |  592,827 |  296 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/kitchen_sink-512c3448ad7ab7b257cd1b2755d8c1cbcdb37b31.md) | 3,787 |  1,979,971 |  889 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/512c3448ad7ab7b257cd1b2755d8c1cbcdb37b31

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29220011686)
