| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/fibonacci-4f7677d07d6cf06531aab96fc2fed6e96387f70e.md) | 3,044 |  12,000,265 |  677 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/keccak-4f7677d07d6cf06531aab96fc2fed6e96387f70e.md) | 16,690 |  18,655,329 |  3,085 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/sha2_bench-4f7677d07d6cf06531aab96fc2fed6e96387f70e.md) | 9,225 |  14,793,960 |  1,129 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/regex-4f7677d07d6cf06531aab96fc2fed6e96387f70e.md) | 1,157 |  4,137,067 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/ecrecover-4f7677d07d6cf06531aab96fc2fed6e96387f70e.md) | 600 |  123,583 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/pairing-4f7677d07d6cf06531aab96fc2fed6e96387f70e.md) | 935 |  1,745,757 |  305 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/kitchen_sink-4f7677d07d6cf06531aab96fc2fed6e96387f70e.md) | 4,071 |  2,579,903 |  867 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/fibonacci_e2e-4f7677d07d6cf06531aab96fc2fed6e96387f70e.md) | 1,372 |  12,000,265 |  286 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/regex_e2e-4f7677d07d6cf06531aab96fc2fed6e96387f70e.md) | 620 |  4,137,067 |  167 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/ecrecover_e2e-4f7677d07d6cf06531aab96fc2fed6e96387f70e.md) | 365 |  123,583 |  142 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/pairing_e2e-4f7677d07d6cf06531aab96fc2fed6e96387f70e.md) | 503 |  1,745,757 |  148 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/kitchen_sink_e2e-4f7677d07d6cf06531aab96fc2fed6e96387f70e.md) | 2,168 |  2,579,903 |  382 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4f7677d07d6cf06531aab96fc2fed6e96387f70e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27549416336)
