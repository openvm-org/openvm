| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/fibonacci-efdcfcd6bcc4b63f62aa22744f025827442e5cd3.md) | 3,844 |  12,000,265 |  953 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/keccak-efdcfcd6bcc4b63f62aa22744f025827442e5cd3.md) | 18,624 |  18,655,329 |  3,308 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/sha2_bench-efdcfcd6bcc4b63f62aa22744f025827442e5cd3.md) | 9,045 |  14,793,960 |  1,399 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/regex-efdcfcd6bcc4b63f62aa22744f025827442e5cd3.md) | 1,409 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/ecrecover-efdcfcd6bcc4b63f62aa22744f025827442e5cd3.md) | 638 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/pairing-efdcfcd6bcc4b63f62aa22744f025827442e5cd3.md) | 889 |  1,745,757 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2769/kitchen_sink-efdcfcd6bcc4b63f62aa22744f025827442e5cd3.md) | 2,097 |  2,579,903 |  433 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/efdcfcd6bcc4b63f62aa22744f025827442e5cd3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25234995849)
