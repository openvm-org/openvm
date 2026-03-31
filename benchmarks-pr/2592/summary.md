| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-7add8bc5fe0c069a720e819a86c0dc2f50ab5d96.md) | 3,791 |  12,000,265 |  928 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-7add8bc5fe0c069a720e819a86c0dc2f50ab5d96.md) | 18,613 |  18,655,329 |  3,374 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-7add8bc5fe0c069a720e819a86c0dc2f50ab5d96.md) | 1,435 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-7add8bc5fe0c069a720e819a86c0dc2f50ab5d96.md) | 644 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-7add8bc5fe0c069a720e819a86c0dc2f50ab5d96.md) | 917 |  1,745,757 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-7add8bc5fe0c069a720e819a86c0dc2f50ab5d96.md) | 2,260 |  2,579,903 |  442 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7add8bc5fe0c069a720e819a86c0dc2f50ab5d96

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23819901965)
