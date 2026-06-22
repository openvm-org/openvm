| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/fibonacci-90cdb88711e40f0fbbb4f199c3771d4f1c34d0c3.md) | 1,024 |  4,000,051 |  390 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/keccak-90cdb88711e40f0fbbb4f199c3771d4f1c34d0c3.md) | 16,333 |  14,365,133 |  3,022 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/sha2_bench-90cdb88711e40f0fbbb4f199c3771d4f1c34d0c3.md) | 8,283 |  11,167,961 |  1,007 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/regex-90cdb88711e40f0fbbb4f199c3771d4f1c34d0c3.md) | 1,211 |  4,090,656 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/ecrecover-90cdb88711e40f0fbbb4f199c3771d4f1c34d0c3.md) | 429 |  112,210 |  279 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/pairing-90cdb88711e40f0fbbb4f199c3771d4f1c34d0c3.md) | 594 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/kitchen_sink-90cdb88711e40f0fbbb4f199c3771d4f1c34d0c3.md) | 3,859 |  1,979,971 |  849 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/90cdb88711e40f0fbbb4f199c3771d4f1c34d0c3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27980724144)
