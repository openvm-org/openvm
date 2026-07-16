| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/fibonacci-e38e82696a82195864e7efd4e54d78d106476919.md) | 409 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/keccak-e38e82696a82195864e7efd4e54d78d106476919.md) | 8,583 |  14,365,133 |  1,514 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/sha2_bench-e38e82696a82195864e7efd4e54d78d106476919.md) | 4,226 |  11,167,961 |  528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/regex-e38e82696a82195864e7efd4e54d78d106476919.md) | 566 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/ecrecover-e38e82696a82195864e7efd4e54d78d106476919.md) | 223 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/pairing-e38e82696a82195864e7efd4e54d78d106476919.md) | 285 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/kitchen_sink-e38e82696a82195864e7efd4e54d78d106476919.md) | 1,930 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e38e82696a82195864e7efd4e54d78d106476919

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29517995569)
