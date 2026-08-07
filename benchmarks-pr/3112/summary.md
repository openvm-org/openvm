| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/fibonacci-bafd483a3d073f9577df88a169a8817f99c31ec9.md) | 441 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/keccak-bafd483a3d073f9577df88a169a8817f99c31ec9.md) | 7,096 |  14,365,133 |  1,498 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/sha2_bench-bafd483a3d073f9577df88a169a8817f99c31ec9.md) | 4,059 |  11,167,961 |  516 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/regex-bafd483a3d073f9577df88a169a8817f99c31ec9.md) | 711 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/ecrecover-bafd483a3d073f9577df88a169a8817f99c31ec9.md) | 203 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/pairing-bafd483a3d073f9577df88a169a8817f99c31ec9.md) | 237 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/kitchen_sink-bafd483a3d073f9577df88a169a8817f99c31ec9.md) | 2,177 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bafd483a3d073f9577df88a169a8817f99c31ec9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31193699663)
