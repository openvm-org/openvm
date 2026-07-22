| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/fibonacci-edc4b7941313ee8e0cb80304357ec6402be5d457.md) | 475 |  4,000,051 |  243 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/keccak-edc4b7941313ee8e0cb80304357ec6402be5d457.md) | 7,329 |  14,365,133 |  1,543 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/sha2_bench-edc4b7941313ee8e0cb80304357ec6402be5d457.md) | 4,679 |  11,167,961 |  535 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/regex-edc4b7941313ee8e0cb80304357ec6402be5d457.md) | 672 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/ecrecover-edc4b7941313ee8e0cb80304357ec6402be5d457.md) | 228 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/pairing-edc4b7941313ee8e0cb80304357ec6402be5d457.md) | 317 |  592,827 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/kitchen_sink-edc4b7941313ee8e0cb80304357ec6402be5d457.md) | 2,668 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/edc4b7941313ee8e0cb80304357ec6402be5d457

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29956102445)
