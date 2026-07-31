| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-9e5234222c3fc3965d51941ed166b3da518dc318.md) | 1,580 |  12,000,265 |  364 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-9e5234222c3fc3965d51941ed166b3da518dc318.md) | 9,314 |  18,655,329 |  1,523 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-9e5234222c3fc3965d51941ed166b3da518dc318.md) | 4,915 |  14,793,960 |  577 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-9e5234222c3fc3965d51941ed166b3da518dc318.md) | 661 |  4,137,067 |  211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-9e5234222c3fc3965d51941ed166b3da518dc318.md) | 432 |  123,583 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-9e5234222c3fc3965d51941ed166b3da518dc318.md) | 560 |  1,745,757 |  191 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-9e5234222c3fc3965d51941ed166b3da518dc318.md) | 2,219 |  2,579,903 |  474 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9e5234222c3fc3965d51941ed166b3da518dc318

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30649880592)
