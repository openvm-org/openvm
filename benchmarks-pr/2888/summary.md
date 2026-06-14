| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/fibonacci-4f7677d07d6cf06531aab96fc2fed6e96387f70e.md) | 3,083 |  12,000,265 |  678 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/keccak-4f7677d07d6cf06531aab96fc2fed6e96387f70e.md) | 16,658 |  18,655,329 |  3,089 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/sha2_bench-4f7677d07d6cf06531aab96fc2fed6e96387f70e.md) | 9,129 |  14,793,960 |  1,118 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/regex-4f7677d07d6cf06531aab96fc2fed6e96387f70e.md) | 1,182 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/ecrecover-4f7677d07d6cf06531aab96fc2fed6e96387f70e.md) | 600 |  123,583 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/pairing-4f7677d07d6cf06531aab96fc2fed6e96387f70e.md) | 938 |  1,745,757 |  305 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/kitchen_sink-4f7677d07d6cf06531aab96fc2fed6e96387f70e.md) | 4,067 |  2,579,903 |  872 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4f7677d07d6cf06531aab96fc2fed6e96387f70e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27495962860)
