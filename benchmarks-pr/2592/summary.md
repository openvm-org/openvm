| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-186d69191f778bd577e199ae12d8ff1e45e24332.md) | 3,839 |  12,000,265 |  958 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-186d69191f778bd577e199ae12d8ff1e45e24332.md) | 18,384 |  18,655,329 |  3,301 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-186d69191f778bd577e199ae12d8ff1e45e24332.md) | 1,430 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-186d69191f778bd577e199ae12d8ff1e45e24332.md) | 648 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-186d69191f778bd577e199ae12d8ff1e45e24332.md) | 908 |  1,745,757 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-186d69191f778bd577e199ae12d8ff1e45e24332.md) | 2,162 |  2,579,903 |  440 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/186d69191f778bd577e199ae12d8ff1e45e24332

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24208181072)
