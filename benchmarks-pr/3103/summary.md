| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-af4f61dcf5dc1cf5e9ae63618929eb1951a5cede.md) | 474 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-af4f61dcf5dc1cf5e9ae63618929eb1951a5cede.md) | 7,414 |  14,365,133 |  1,533 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-af4f61dcf5dc1cf5e9ae63618929eb1951a5cede.md) | 4,188 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-af4f61dcf5dc1cf5e9ae63618929eb1951a5cede.md) | 676 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-af4f61dcf5dc1cf5e9ae63618929eb1951a5cede.md) | 195 |  112,210 |  197 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-af4f61dcf5dc1cf5e9ae63618929eb1951a5cede.md) | 239 |  592,827 |  197 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-af4f61dcf5dc1cf5e9ae63618929eb1951a5cede.md) | 2,028 |  1,979,971 |  525 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/af4f61dcf5dc1cf5e9ae63618929eb1951a5cede

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31746659623)
