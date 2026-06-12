| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/fibonacci-e7c1354ae4ec90f3265b5d0e86d4871719649c37.md) | 1,675 |  4,000,051 |  534 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/keccak-e7c1354ae4ec90f3265b5d0e86d4871719649c37.md) | 16,428 |  14,365,133 |  3,048 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/sha2_bench-e7c1354ae4ec90f3265b5d0e86d4871719649c37.md) | 10,428 |  11,167,961 |  1,940 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/regex-e7c1354ae4ec90f3265b5d0e86d4871719649c37.md) | 1,533 |  4,090,656 |  431 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/ecrecover-e7c1354ae4ec90f3265b5d0e86d4871719649c37.md) | 488 |  112,210 |  312 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/pairing-e7c1354ae4ec90f3265b5d0e86d4871719649c37.md) | 627 |  592,827 |  300 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/kitchen_sink-e7c1354ae4ec90f3265b5d0e86d4871719649c37.md) | 3,928 |  1,979,971 |  854 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e7c1354ae4ec90f3265b5d0e86d4871719649c37

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27435006398)
