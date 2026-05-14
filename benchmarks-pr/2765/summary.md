| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-9d94b5e8f80803ad33cdf2b28007475bdafe6f45.md) | 1,922 |  4,000,051 |  540 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-9d94b5e8f80803ad33cdf2b28007475bdafe6f45.md) | 13,414 |  14,365,133 |  2,199 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-9d94b5e8f80803ad33cdf2b28007475bdafe6f45.md) | 9,364 |  11,167,961 |  1,401 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-9d94b5e8f80803ad33cdf2b28007475bdafe6f45.md) | 1,579 |  4,090,656 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-9d94b5e8f80803ad33cdf2b28007475bdafe6f45.md) | 639 |  112,210 |  289 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-9d94b5e8f80803ad33cdf2b28007475bdafe6f45.md) | 751 |  592,827 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-9d94b5e8f80803ad33cdf2b28007475bdafe6f45.md) | 2,032 |  1,979,971 |  430 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9d94b5e8f80803ad33cdf2b28007475bdafe6f45

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25853042033)
