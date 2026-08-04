| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/fibonacci-3ddf48bcce410106210fc75cf22aed71d092e5b9.md) | 479 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/keccak-3ddf48bcce410106210fc75cf22aed71d092e5b9.md) | 7,373 |  14,365,133 |  1,524 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/sha2_bench-3ddf48bcce410106210fc75cf22aed71d092e5b9.md) | 4,151 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/regex-3ddf48bcce410106210fc75cf22aed71d092e5b9.md) | 656 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/ecrecover-3ddf48bcce410106210fc75cf22aed71d092e5b9.md) | 228 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/pairing-3ddf48bcce410106210fc75cf22aed71d092e5b9.md) | 236 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/kitchen_sink-3ddf48bcce410106210fc75cf22aed71d092e5b9.md) | 2,042 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3ddf48bcce410106210fc75cf22aed71d092e5b9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30887212165)
