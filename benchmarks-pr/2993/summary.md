| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/fibonacci-e45f223588301151501721c6ac22d2c0f2e5f3ae.md) | 876 |  4,000,051 |  395 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/keccak-e45f223588301151501721c6ac22d2c0f2e5f3ae.md) | 15,685 |  14,365,133 |  3,050 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/sha2_bench-e45f223588301151501721c6ac22d2c0f2e5f3ae.md) | 7,962 |  11,167,961 |  1,004 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/regex-e45f223588301151501721c6ac22d2c0f2e5f3ae.md) | 1,058 |  4,090,656 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/ecrecover-e45f223588301151501721c6ac22d2c0f2e5f3ae.md) | 312 |  112,210 |  286 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/pairing-e45f223588301151501721c6ac22d2c0f2e5f3ae.md) | 451 |  592,827 |  311 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/kitchen_sink-e45f223588301151501721c6ac22d2c0f2e5f3ae.md) | 3,717 |  1,979,971 |  862 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e45f223588301151501721c6ac22d2c0f2e5f3ae

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29107590189)
