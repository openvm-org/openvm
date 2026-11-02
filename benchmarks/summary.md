| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-b8bb0fc5a4c6818973402c6c7137d4602ea71624.md) | 1,697 |  12,000,265 |  377 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-b8bb0fc5a4c6818973402c6c7137d4602ea71624.md) | 9,582 |  18,655,329 |  1,547 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-b8bb0fc5a4c6818973402c6c7137d4602ea71624.md) | 5,243 |  14,793,960 |  586 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-b8bb0fc5a4c6818973402c6c7137d4602ea71624.md) | 688 |  4,137,067 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-b8bb0fc5a4c6818973402c6c7137d4602ea71624.md) | 443 |  123,583 |  192 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-b8bb0fc5a4c6818973402c6c7137d4602ea71624.md) | 572 |  1,745,757 |  192 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-b8bb0fc5a4c6818973402c6c7137d4602ea71624.md) | 2,307 |  2,579,903 |  495 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b8bb0fc5a4c6818973402c6c7137d4602ea71624

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/41529494206)
