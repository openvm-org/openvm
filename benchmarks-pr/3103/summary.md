| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-b0f349d3c2b01864e371ae25c6d38c6499b1780a.md) | 449 |  4,000,051 |  227 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-b0f349d3c2b01864e371ae25c6d38c6499b1780a.md) | 7,435 |  14,365,133 |  1,529 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-b0f349d3c2b01864e371ae25c6d38c6499b1780a.md) | 4,131 |  11,167,961 |  513 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-b0f349d3c2b01864e371ae25c6d38c6499b1780a.md) | 668 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-b0f349d3c2b01864e371ae25c6d38c6499b1780a.md) | 199 |  112,210 |  199 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-b0f349d3c2b01864e371ae25c6d38c6499b1780a.md) | 235 |  592,827 |  195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-b0f349d3c2b01864e371ae25c6d38c6499b1780a.md) | 2,014 |  1,979,971 |  526 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b0f349d3c2b01864e371ae25c6d38c6499b1780a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31694468624)
