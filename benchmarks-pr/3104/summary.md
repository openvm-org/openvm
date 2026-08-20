| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-00ffe5c889f5000bf2b42b3d70ea02888114d17b.md) | 436 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-00ffe5c889f5000bf2b42b3d70ea02888114d17b.md) | 7,198 |  14,365,133 |  1,576 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-00ffe5c889f5000bf2b42b3d70ea02888114d17b.md) | 4,144 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-00ffe5c889f5000bf2b42b3d70ea02888114d17b.md) | 700 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-00ffe5c889f5000bf2b42b3d70ea02888114d17b.md) | 209 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-00ffe5c889f5000bf2b42b3d70ea02888114d17b.md) | 241 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-00ffe5c889f5000bf2b42b3d70ea02888114d17b.md) | 2,164 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/00ffe5c889f5000bf2b42b3d70ea02888114d17b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32316336301)
