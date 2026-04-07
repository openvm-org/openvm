| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2670/fibonacci-38faf82a8cac444aaf2064d64cf46e91c91393b8.md) | 3,813 |  12,000,265 |  946 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2670/keccak-38faf82a8cac444aaf2064d64cf46e91c91393b8.md) | 18,463 |  18,655,329 |  3,327 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2670/regex-38faf82a8cac444aaf2064d64cf46e91c91393b8.md) | 1,402 |  4,137,067 |  372 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2670/ecrecover-38faf82a8cac444aaf2064d64cf46e91c91393b8.md) | 641 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2670/pairing-38faf82a8cac444aaf2064d64cf46e91c91393b8.md) | 910 |  1,745,757 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2670/kitchen_sink-38faf82a8cac444aaf2064d64cf46e91c91393b8.md) | 2,296 |  2,579,903 |  447 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/38faf82a8cac444aaf2064d64cf46e91c91393b8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24104998659)
