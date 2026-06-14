| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/fibonacci-750bc4073dddf79ecc66b8b26ce47674cc3f1226.md) | 3,085 |  12,000,265 |  679 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/keccak-750bc4073dddf79ecc66b8b26ce47674cc3f1226.md) | 16,203 |  18,655,329 |  3,000 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/sha2_bench-750bc4073dddf79ecc66b8b26ce47674cc3f1226.md) | 9,158 |  14,793,960 |  1,122 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/regex-750bc4073dddf79ecc66b8b26ce47674cc3f1226.md) | 1,170 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/ecrecover-750bc4073dddf79ecc66b8b26ce47674cc3f1226.md) | 607 |  123,583 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/pairing-750bc4073dddf79ecc66b8b26ce47674cc3f1226.md) | 936 |  1,745,757 |  309 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/kitchen_sink-750bc4073dddf79ecc66b8b26ce47674cc3f1226.md) | 4,102 |  2,579,903 |  880 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/750bc4073dddf79ecc66b8b26ce47674cc3f1226

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27493635257)
