| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-4df0fa8af3cc5ef6e546e4b1fa18158c0a91bd7f.md) | 891 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-4df0fa8af3cc5ef6e546e4b1fa18158c0a91bd7f.md) | 8,555 |  14,365,133 |  1,508 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-4df0fa8af3cc5ef6e546e4b1fa18158c0a91bd7f.md) | 4,184 |  11,167,961 |  513 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-4df0fa8af3cc5ef6e546e4b1fa18158c0a91bd7f.md) | 736 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-4df0fa8af3cc5ef6e546e4b1fa18158c0a91bd7f.md) | 498 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-4df0fa8af3cc5ef6e546e4b1fa18158c0a91bd7f.md) | 476 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-4df0fa8af3cc5ef6e546e4b1fa18158c0a91bd7f.md) | 2,352 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4df0fa8af3cc5ef6e546e4b1fa18158c0a91bd7f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31022383256)
