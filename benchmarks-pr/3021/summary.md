| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3021/fibonacci-ddc2198e0025779f89690a7d3f6b2dfb62f8afcc.md) | 410 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3021/keccak-ddc2198e0025779f89690a7d3f6b2dfb62f8afcc.md) | 8,366 |  14,365,133 |  1,513 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3021/sha2_bench-ddc2198e0025779f89690a7d3f6b2dfb62f8afcc.md) | 3,994 |  11,167,961 |  530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3021/regex-ddc2198e0025779f89690a7d3f6b2dfb62f8afcc.md) | 570 |  4,090,656 |  211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3021/ecrecover-ddc2198e0025779f89690a7d3f6b2dfb62f8afcc.md) | 217 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3021/pairing-ddc2198e0025779f89690a7d3f6b2dfb62f8afcc.md) | 261 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3021/kitchen_sink-ddc2198e0025779f89690a7d3f6b2dfb62f8afcc.md) | 1,884 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ddc2198e0025779f89690a7d3f6b2dfb62f8afcc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29423054562)
