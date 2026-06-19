| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2915/fibonacci-5856a783c9a5fd1e4c04082ddb42841b34733fd4.md) | 3,080 |  12,000,265 |  678 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2915/keccak-5856a783c9a5fd1e4c04082ddb42841b34733fd4.md) | 16,610 |  18,655,329 |  3,096 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2915/sha2_bench-5856a783c9a5fd1e4c04082ddb42841b34733fd4.md) | 9,174 |  14,793,960 |  1,121 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2915/regex-5856a783c9a5fd1e4c04082ddb42841b34733fd4.md) | 1,176 |  4,137,067 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2915/ecrecover-5856a783c9a5fd1e4c04082ddb42841b34733fd4.md) | 603 |  123,583 |  285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2915/pairing-5856a783c9a5fd1e4c04082ddb42841b34733fd4.md) | 937 |  1,745,757 |  307 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2915/kitchen_sink-5856a783c9a5fd1e4c04082ddb42841b34733fd4.md) | 4,117 |  2,579,903 |  882 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5856a783c9a5fd1e4c04082ddb42841b34733fd4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27849977100)
