| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-0c6266de4754dd68bcea87508f219429eca1cf99.md) | 1,656 |  12,000,265 |  367 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-0c6266de4754dd68bcea87508f219429eca1cf99.md) | 9,609 |  18,655,329 |  1,538 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-0c6266de4754dd68bcea87508f219429eca1cf99.md) | 5,261 |  14,793,960 |  593 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-0c6266de4754dd68bcea87508f219429eca1cf99.md) | 714 |  4,137,067 |  220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-0c6266de4754dd68bcea87508f219429eca1cf99.md) | 435 |  123,583 |  191 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-0c6266de4754dd68bcea87508f219429eca1cf99.md) | 574 |  1,745,757 |  195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-0c6266de4754dd68bcea87508f219429eca1cf99.md) | 2,290 |  2,579,903 |  497 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0c6266de4754dd68bcea87508f219429eca1cf99

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33098740713)
