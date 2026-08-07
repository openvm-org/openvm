| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-bafd483a3d073f9577df88a169a8817f99c31ec9.md) | 443 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-bafd483a3d073f9577df88a169a8817f99c31ec9.md) | 7,123 |  14,365,133 |  1,497 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-bafd483a3d073f9577df88a169a8817f99c31ec9.md) | 4,109 |  11,167,961 |  519 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-bafd483a3d073f9577df88a169a8817f99c31ec9.md) | 714 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-bafd483a3d073f9577df88a169a8817f99c31ec9.md) | 201 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-bafd483a3d073f9577df88a169a8817f99c31ec9.md) | 239 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-bafd483a3d073f9577df88a169a8817f99c31ec9.md) | 2,157 |  1,979,971 |  456 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bafd483a3d073f9577df88a169a8817f99c31ec9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31133634724)
