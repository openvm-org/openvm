| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/fibonacci-f49e1598ed6bf0f4fe9934a0ef1e429e0c9c898e.md) | 3,779 |  12,000,265 |  925 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/keccak-f49e1598ed6bf0f4fe9934a0ef1e429e0c9c898e.md) | 18,736 |  18,655,329 |  3,303 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/sha2_bench-f49e1598ed6bf0f4fe9934a0ef1e429e0c9c898e.md) | 10,165 |  14,793,960 |  1,445 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/regex-f49e1598ed6bf0f4fe9934a0ef1e429e0c9c898e.md) | 1,403 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/ecrecover-f49e1598ed6bf0f4fe9934a0ef1e429e0c9c898e.md) | 603 |  123,583 |  252 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/pairing-f49e1598ed6bf0f4fe9934a0ef1e429e0c9c898e.md) | 897 |  1,745,757 |  265 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/kitchen_sink-f49e1598ed6bf0f4fe9934a0ef1e429e0c9c898e.md) | 1,902 |  2,579,903 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f49e1598ed6bf0f4fe9934a0ef1e429e0c9c898e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26258456340)
