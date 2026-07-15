| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3005/fibonacci-673fc01a1a555aab4f1afacc1590e78eb120ef64.md) | 462 |  4,000,051 |  226 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3005/keccak-673fc01a1a555aab4f1afacc1590e78eb120ef64.md) | 8,675 |  14,365,133 |  1,518 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3005/sha2_bench-673fc01a1a555aab4f1afacc1590e78eb120ef64.md) | 3,950 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3005/regex-673fc01a1a555aab4f1afacc1590e78eb120ef64.md) | 505 |  4,090,656 |  191 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3005/ecrecover-673fc01a1a555aab4f1afacc1590e78eb120ef64.md) | 221 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3005/pairing-673fc01a1a555aab4f1afacc1590e78eb120ef64.md) | 264 |  592,827 |  180 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3005/kitchen_sink-673fc01a1a555aab4f1afacc1590e78eb120ef64.md) | 1,931 |  1,979,971 |  466 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/673fc01a1a555aab4f1afacc1590e78eb120ef64

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29410088140)
