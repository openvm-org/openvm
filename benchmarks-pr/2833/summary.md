| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/fibonacci-51a16b45edc009c1f5c8ebc00256594105753b7d.md) | 1,023 |  4,000,051 |  392 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/keccak-51a16b45edc009c1f5c8ebc00256594105753b7d.md) | 16,473 |  14,365,133 |  3,061 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/sha2_bench-51a16b45edc009c1f5c8ebc00256594105753b7d.md) | 8,105 |  11,167,961 |  988 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/regex-51a16b45edc009c1f5c8ebc00256594105753b7d.md) | 1,223 |  4,090,656 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/ecrecover-51a16b45edc009c1f5c8ebc00256594105753b7d.md) | 434 |  112,210 |  288 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/pairing-51a16b45edc009c1f5c8ebc00256594105753b7d.md) | 600 |  592,827 |  296 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/kitchen_sink-51a16b45edc009c1f5c8ebc00256594105753b7d.md) | 3,869 |  1,979,971 |  854 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/51a16b45edc009c1f5c8ebc00256594105753b7d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27843636318)
