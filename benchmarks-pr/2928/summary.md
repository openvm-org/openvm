| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2928/fibonacci-a79812ebbb8434f802903e28cb0083443efacb60.md) | 1,028 |  4,000,051 |  392 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2928/keccak-a79812ebbb8434f802903e28cb0083443efacb60.md) | 16,320 |  14,365,133 |  3,015 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2928/sha2_bench-a79812ebbb8434f802903e28cb0083443efacb60.md) | 8,264 |  11,167,961 |  999 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2928/regex-a79812ebbb8434f802903e28cb0083443efacb60.md) | 1,225 |  4,090,656 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2928/ecrecover-a79812ebbb8434f802903e28cb0083443efacb60.md) | 436 |  112,210 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2928/pairing-a79812ebbb8434f802903e28cb0083443efacb60.md) | 610 |  592,827 |  298 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2928/kitchen_sink-a79812ebbb8434f802903e28cb0083443efacb60.md) | 3,898 |  1,979,971 |  857 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a79812ebbb8434f802903e28cb0083443efacb60

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28242946833)
