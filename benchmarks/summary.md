| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-af635f6bafb5d3ba0d7d9b3892b24ec76f3365f0.md) | 1,659 |  12,000,265 |  369 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-af635f6bafb5d3ba0d7d9b3892b24ec76f3365f0.md) | 9,669 |  18,655,329 |  1,545 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-af635f6bafb5d3ba0d7d9b3892b24ec76f3365f0.md) | 5,271 |  14,793,960 |  587 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-af635f6bafb5d3ba0d7d9b3892b24ec76f3365f0.md) | 704 |  4,137,067 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-af635f6bafb5d3ba0d7d9b3892b24ec76f3365f0.md) | 432 |  123,583 |  194 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-af635f6bafb5d3ba0d7d9b3892b24ec76f3365f0.md) | 584 |  1,745,757 |  194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-af635f6bafb5d3ba0d7d9b3892b24ec76f3365f0.md) | 2,300 |  2,579,903 |  492 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/af635f6bafb5d3ba0d7d9b3892b24ec76f3365f0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33104000082)
