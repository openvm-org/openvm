| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/fibonacci-dce9e3a2c6b4e9ef98b9badf4cf0b8b7f07b7c00.md) | 3,065 |  12,000,265 |  676 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/keccak-dce9e3a2c6b4e9ef98b9badf4cf0b8b7f07b7c00.md) | 16,386 |  18,655,329 |  3,019 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/sha2_bench-dce9e3a2c6b4e9ef98b9badf4cf0b8b7f07b7c00.md) | 9,149 |  14,793,960 |  1,125 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/regex-dce9e3a2c6b4e9ef98b9badf4cf0b8b7f07b7c00.md) | 1,162 |  4,137,067 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/ecrecover-dce9e3a2c6b4e9ef98b9badf4cf0b8b7f07b7c00.md) | 597 |  123,583 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/pairing-dce9e3a2c6b4e9ef98b9badf4cf0b8b7f07b7c00.md) | 947 |  1,745,757 |  307 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/kitchen_sink-dce9e3a2c6b4e9ef98b9badf4cf0b8b7f07b7c00.md) | 4,126 |  2,579,903 |  878 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/dce9e3a2c6b4e9ef98b9badf4cf0b8b7f07b7c00

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27975214726)
