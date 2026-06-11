| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/fibonacci-2b6ac52899e0de6e232799e40f1f1ed20fa7467e.md) | 5,420 |  4,000,051 |  540 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/keccak-2b6ac52899e0de6e232799e40f1f1ed20fa7467e.md) | 20,295 |  14,365,133 |  3,043 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/sha2_bench-2b6ac52899e0de6e232799e40f1f1ed20fa7467e.md) | 14,048 |  11,167,961 |  1,928 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/regex-2b6ac52899e0de6e232799e40f1f1ed20fa7467e.md) | 3,767 |  4,090,656 |  428 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/ecrecover-2b6ac52899e0de6e232799e40f1f1ed20fa7467e.md) | 1,968 |  112,210 |  309 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/pairing-2b6ac52899e0de6e232799e40f1f1ed20fa7467e.md) | 2,085 |  592,827 |  296 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/kitchen_sink-2b6ac52899e0de6e232799e40f1f1ed20fa7467e.md) | 5,680 |  1,979,971 |  867 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2b6ac52899e0de6e232799e40f1f1ed20fa7467e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27356042997)
