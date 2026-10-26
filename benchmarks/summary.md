| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-91f93b8f43ccc7fa615cf6aeb9b1f278b5ab9e8f.md) | 1,689 |  12,000,265 |  377 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-91f93b8f43ccc7fa615cf6aeb9b1f278b5ab9e8f.md) | 9,672 |  18,655,329 |  1,548 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-91f93b8f43ccc7fa615cf6aeb9b1f278b5ab9e8f.md) | 5,224 |  14,793,960 |  587 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-91f93b8f43ccc7fa615cf6aeb9b1f278b5ab9e8f.md) | 703 |  4,137,067 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-91f93b8f43ccc7fa615cf6aeb9b1f278b5ab9e8f.md) | 437 |  123,583 |  189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-91f93b8f43ccc7fa615cf6aeb9b1f278b5ab9e8f.md) | 590 |  1,745,757 |  195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-91f93b8f43ccc7fa615cf6aeb9b1f278b5ab9e8f.md) | 2,282 |  2,579,903 |  491 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/91f93b8f43ccc7fa615cf6aeb9b1f278b5ab9e8f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/40389009777)
