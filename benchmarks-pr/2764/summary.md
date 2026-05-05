| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/fibonacci-dc2352fa015d8f0c714597c9091f5f98d4b70c03.md) | 3,892 |  12,000,265 |  974 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/keccak-dc2352fa015d8f0c714597c9091f5f98d4b70c03.md) | 19,081 |  18,655,329 |  3,355 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/sha2_bench-dc2352fa015d8f0c714597c9091f5f98d4b70c03.md) | 9,060 |  14,793,960 |  1,394 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/regex-dc2352fa015d8f0c714597c9091f5f98d4b70c03.md) | 1,421 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/ecrecover-dc2352fa015d8f0c714597c9091f5f98d4b70c03.md) | 639 |  123,583 |  278 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/pairing-dc2352fa015d8f0c714597c9091f5f98d4b70c03.md) | 911 |  1,745,757 |  287 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/kitchen_sink-dc2352fa015d8f0c714597c9091f5f98d4b70c03.md) | 2,035 |  2,579,903 |  433 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/dc2352fa015d8f0c714597c9091f5f98d4b70c03

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25405735580)
