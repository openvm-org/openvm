| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2887/fibonacci-4813b4b58bb674072996a7e2919e0e8bb0f9e395.md) | 1,645 |  4,000,051 |  521 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2887/keccak-4813b4b58bb674072996a7e2919e0e8bb0f9e395.md) | 16,359 |  14,365,133 |  3,052 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2887/sha2_bench-4813b4b58bb674072996a7e2919e0e8bb0f9e395.md) | 10,447 |  11,167,961 |  1,940 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2887/regex-4813b4b58bb674072996a7e2919e0e8bb0f9e395.md) | 1,545 |  4,090,656 |  430 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2887/ecrecover-4813b4b58bb674072996a7e2919e0e8bb0f9e395.md) | 486 |  112,210 |  307 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2887/pairing-4813b4b58bb674072996a7e2919e0e8bb0f9e395.md) | 632 |  592,827 |  308 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2887/kitchen_sink-4813b4b58bb674072996a7e2919e0e8bb0f9e395.md) | 3,915 |  1,979,971 |  860 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4813b4b58bb674072996a7e2919e0e8bb0f9e395

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27436481886)
