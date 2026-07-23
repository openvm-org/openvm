| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-c83a208a3e51c6520fb1771d60e9e52c04b9067b.md) | 483 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-c83a208a3e51c6520fb1771d60e9e52c04b9067b.md) | 7,323 |  14,365,133 |  1,530 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-c83a208a3e51c6520fb1771d60e9e52c04b9067b.md) | 4,738 |  11,167,961 |  528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-c83a208a3e51c6520fb1771d60e9e52c04b9067b.md) | 672 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-c83a208a3e51c6520fb1771d60e9e52c04b9067b.md) | 227 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-c83a208a3e51c6520fb1771d60e9e52c04b9067b.md) | 274 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-c83a208a3e51c6520fb1771d60e9e52c04b9067b.md) | 2,736 |  1,979,971 |  466 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c83a208a3e51c6520fb1771d60e9e52c04b9067b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29971264968)
