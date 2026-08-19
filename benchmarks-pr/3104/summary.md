| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-943128db715d9d10eed29f93fc66a9a820fc2164.md) | 449 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-943128db715d9d10eed29f93fc66a9a820fc2164.md) | 7,321 |  14,365,133 |  1,638 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-943128db715d9d10eed29f93fc66a9a820fc2164.md) | 4,138 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-943128db715d9d10eed29f93fc66a9a820fc2164.md) | 701 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-943128db715d9d10eed29f93fc66a9a820fc2164.md) | 208 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-943128db715d9d10eed29f93fc66a9a820fc2164.md) | 241 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-943128db715d9d10eed29f93fc66a9a820fc2164.md) | 2,179 |  1,979,971 |  464 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/943128db715d9d10eed29f93fc66a9a820fc2164

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32308693563)
