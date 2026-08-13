| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/fibonacci-4ba087cd860326e246322001b3a34506f3fda759.md) | 445 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/keccak-4ba087cd860326e246322001b3a34506f3fda759.md) | 7,158 |  14,365,133 |  1,578 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/sha2_bench-4ba087cd860326e246322001b3a34506f3fda759.md) | 4,079 |  11,167,961 |  513 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/regex-4ba087cd860326e246322001b3a34506f3fda759.md) | 726 |  4,090,656 |  222 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/ecrecover-4ba087cd860326e246322001b3a34506f3fda759.md) | 203 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/pairing-4ba087cd860326e246322001b3a34506f3fda759.md) | 247 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/kitchen_sink-4ba087cd860326e246322001b3a34506f3fda759.md) | 2,179 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4ba087cd860326e246322001b3a34506f3fda759

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31720214622)
