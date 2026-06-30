| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/fibonacci-3b2c48f329b6d218c7522b9d399bd4e0b3a89847.md) | 859 |  4,000,051 |  388 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/keccak-3b2c48f329b6d218c7522b9d399bd4e0b3a89847.md) | 15,347 |  14,365,133 |  3,008 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/sha2_bench-3b2c48f329b6d218c7522b9d399bd4e0b3a89847.md) | 8,023 |  11,167,961 |  1,000 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/regex-3b2c48f329b6d218c7522b9d399bd4e0b3a89847.md) | 1,029 |  4,090,656 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/ecrecover-3b2c48f329b6d218c7522b9d399bd4e0b3a89847.md) | 302 |  112,210 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/pairing-3b2c48f329b6d218c7522b9d399bd4e0b3a89847.md) | 453 |  592,827 |  297 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/kitchen_sink-3b2c48f329b6d218c7522b9d399bd4e0b3a89847.md) | 3,729 |  1,979,971 |  857 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3b2c48f329b6d218c7522b9d399bd4e0b3a89847

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28471908341)
