| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/fibonacci-af87543ddac05da5e715f87fd092528fdcad1098.md) | 1,028 |  4,000,051 |  389 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/keccak-af87543ddac05da5e715f87fd092528fdcad1098.md) | 16,077 |  14,365,133 |  2,983 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/sha2_bench-af87543ddac05da5e715f87fd092528fdcad1098.md) | 8,474 |  11,167,961 |  1,024 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/regex-af87543ddac05da5e715f87fd092528fdcad1098.md) | 1,215 |  4,090,656 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/ecrecover-af87543ddac05da5e715f87fd092528fdcad1098.md) | 443 |  112,210 |  286 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/pairing-af87543ddac05da5e715f87fd092528fdcad1098.md) | 598 |  592,827 |  296 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/kitchen_sink-af87543ddac05da5e715f87fd092528fdcad1098.md) | 3,869 |  1,979,971 |  854 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/af87543ddac05da5e715f87fd092528fdcad1098

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28054678819)
