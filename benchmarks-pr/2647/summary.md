| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2647/fibonacci-c03ed4ddbf9cb1c2761086f87ae40296b4fc382d.md) | 3,835 |  12,000,265 |  945 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2647/keccak-c03ed4ddbf9cb1c2761086f87ae40296b4fc382d.md) | 18,474 |  18,655,329 |  3,294 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2647/regex-c03ed4ddbf9cb1c2761086f87ae40296b4fc382d.md) | 1,421 |  4,137,067 |  369 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2647/ecrecover-c03ed4ddbf9cb1c2761086f87ae40296b4fc382d.md) | 733 |  317,792 |  353 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2647/pairing-c03ed4ddbf9cb1c2761086f87ae40296b4fc382d.md) | 910 |  1,745,757 |  312 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2647/kitchen_sink-c03ed4ddbf9cb1c2761086f87ae40296b4fc382d.md) | 2,545 |  2,580,026 |  548 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c03ed4ddbf9cb1c2761086f87ae40296b4fc382d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23867730871)
