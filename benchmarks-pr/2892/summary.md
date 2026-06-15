| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2892/fibonacci-c9942e0e2949e0520f66f4ea31542a885c96c9de.md) | 1,633 |  4,000,051 |  522 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2892/keccak-c9942e0e2949e0520f66f4ea31542a885c96c9de.md) | 16,526 |  14,365,133 |  3,079 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2892/sha2_bench-c9942e0e2949e0520f66f4ea31542a885c96c9de.md) | 10,517 |  11,167,961 |  1,956 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2892/regex-c9942e0e2949e0520f66f4ea31542a885c96c9de.md) | 1,529 |  4,090,656 |  429 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2892/ecrecover-c9942e0e2949e0520f66f4ea31542a885c96c9de.md) | 479 |  112,210 |  311 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2892/pairing-c9942e0e2949e0520f66f4ea31542a885c96c9de.md) | 621 |  592,827 |  293 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2892/kitchen_sink-c9942e0e2949e0520f66f4ea31542a885c96c9de.md) | 3,917 |  1,979,971 |  857 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c9942e0e2949e0520f66f4ea31542a885c96c9de

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27575933908)
