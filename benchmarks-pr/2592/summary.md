| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-4ec78d628590a830eaa8b2896784e2d975071525.md) | 3,758 |  12,000,265 |  915 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-4ec78d628590a830eaa8b2896784e2d975071525.md) | 18,635 |  18,655,329 |  3,300 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-4ec78d628590a830eaa8b2896784e2d975071525.md) | 10,116 |  14,793,960 |  1,447 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-4ec78d628590a830eaa8b2896784e2d975071525.md) | 1,396 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-4ec78d628590a830eaa8b2896784e2d975071525.md) | 599 |  123,583 |  251 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-4ec78d628590a830eaa8b2896784e2d975071525.md) | 889 |  1,745,757 |  266 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-4ec78d628590a830eaa8b2896784e2d975071525.md) | 1,889 |  2,579,903 |  409 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-4ec78d628590a830eaa8b2896784e2d975071525.md) | 1,784 |  12,000,265 |  410 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-4ec78d628590a830eaa8b2896784e2d975071525.md) | 812 |  4,137,067 |  170 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-4ec78d628590a830eaa8b2896784e2d975071525.md) | 512 |  123,583 |  133 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-4ec78d628590a830eaa8b2896784e2d975071525.md) | 634 |  1,745,757 |  131 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-4ec78d628590a830eaa8b2896784e2d975071525.md) | 2,038 |  2,579,903 |  402 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4ec78d628590a830eaa8b2896784e2d975071525

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25887436712)
