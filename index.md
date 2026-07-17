| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-725018c72b50fa6f2eae184516992c968768ed5e.md) | 1,586 |  12,000,265 |  361 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-725018c72b50fa6f2eae184516992c968768ed5e.md) | 9,255 |  18,655,329 |  1,515 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-725018c72b50fa6f2eae184516992c968768ed5e.md) | 4,875 |  14,793,960 |  572 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-725018c72b50fa6f2eae184516992c968768ed5e.md) | 662 |  4,137,067 |  210 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-725018c72b50fa6f2eae184516992c968768ed5e.md) | 427 |  123,583 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-725018c72b50fa6f2eae184516992c968768ed5e.md) | 570 |  1,745,757 |  192 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-725018c72b50fa6f2eae184516992c968768ed5e.md) | 2,214 |  2,579,903 |  477 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/725018c72b50fa6f2eae184516992c968768ed5e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29598884175)
