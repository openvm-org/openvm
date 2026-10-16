| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/fibonacci-0c9d02c6f425b6a6a22aa23fc721659b29068a46.md) | 477 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/keccak-0c9d02c6f425b6a6a22aa23fc721659b29068a46.md) | 7,650 |  14,365,133 |  1,620 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/sha2_bench-0c9d02c6f425b6a6a22aa23fc721659b29068a46.md) | 4,331 |  11,167,961 |  531 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/regex-0c9d02c6f425b6a6a22aa23fc721659b29068a46.md) | 743 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/ecrecover-0c9d02c6f425b6a6a22aa23fc721659b29068a46.md) | 212 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/pairing-0c9d02c6f425b6a6a22aa23fc721659b29068a46.md) | 252 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/kitchen_sink-0c9d02c6f425b6a6a22aa23fc721659b29068a46.md) | 2,258 |  1,979,971 |  476 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0c9d02c6f425b6a6a22aa23fc721659b29068a46

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/38978901161)
