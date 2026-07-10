| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3002/fibonacci-9e2ad10c3733eb2ea5f7e2f41dd8969bf4091e41.md) | 3,008 |  12,000,265 |  677 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3002/keccak-9e2ad10c3733eb2ea5f7e2f41dd8969bf4091e41.md) | 16,528 |  18,655,329 |  3,044 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3002/sha2_bench-9e2ad10c3733eb2ea5f7e2f41dd8969bf4091e41.md) | 9,419 |  14,793,960 |  1,126 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3002/regex-9e2ad10c3733eb2ea5f7e2f41dd8969bf4091e41.md) | 1,215 |  4,137,067 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3002/ecrecover-9e2ad10c3733eb2ea5f7e2f41dd8969bf4091e41.md) | 511 |  123,583 |  278 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3002/pairing-9e2ad10c3733eb2ea5f7e2f41dd8969bf4091e41.md) | 846 |  1,745,757 |  311 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3002/kitchen_sink-9e2ad10c3733eb2ea5f7e2f41dd8969bf4091e41.md) | 4,460 |  2,579,903 |  870 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9e2ad10c3733eb2ea5f7e2f41dd8969bf4091e41

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29076202090)
