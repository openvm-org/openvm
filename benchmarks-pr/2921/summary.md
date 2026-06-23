| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/fibonacci-76ca53c369580b21b4005ff976e8d02af68412a6.md) | 1,018 |  4,000,051 |  387 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/keccak-76ca53c369580b21b4005ff976e8d02af68412a6.md) | 16,457 |  14,365,133 |  3,050 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/sha2_bench-76ca53c369580b21b4005ff976e8d02af68412a6.md) | 8,273 |  11,167,961 |  1,012 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/regex-76ca53c369580b21b4005ff976e8d02af68412a6.md) | 1,222 |  4,090,656 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/ecrecover-76ca53c369580b21b4005ff976e8d02af68412a6.md) | 431 |  112,210 |  276 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/pairing-76ca53c369580b21b4005ff976e8d02af68412a6.md) | 599 |  592,827 |  299 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/kitchen_sink-76ca53c369580b21b4005ff976e8d02af68412a6.md) | 3,851 |  1,979,971 |  845 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/76ca53c369580b21b4005ff976e8d02af68412a6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28052453963)
