| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-e11f7868ca90c7ecf97d975d759211d28e9a4d91.md) | 3,843 |  12,000,265 |  963 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-e11f7868ca90c7ecf97d975d759211d28e9a4d91.md) | 18,934 |  18,655,329 |  3,328 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-e11f7868ca90c7ecf97d975d759211d28e9a4d91.md) | 9,046 |  14,793,960 |  1,398 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-e11f7868ca90c7ecf97d975d759211d28e9a4d91.md) | 1,423 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-e11f7868ca90c7ecf97d975d759211d28e9a4d91.md) | 630 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-e11f7868ca90c7ecf97d975d759211d28e9a4d91.md) | 904 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-e11f7868ca90c7ecf97d975d759211d28e9a4d91.md) | 2,057 |  2,579,903 |  438 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-e11f7868ca90c7ecf97d975d759211d28e9a4d91.md) | 1,828 |  12,000,265 |  452 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-e11f7868ca90c7ecf97d975d759211d28e9a4d91.md) | 846 |  4,137,067 |  192 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-e11f7868ca90c7ecf97d975d759211d28e9a4d91.md) | 545 |  123,583 |  151 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-e11f7868ca90c7ecf97d975d759211d28e9a4d91.md) | 651 |  1,745,757 |  152 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-e11f7868ca90c7ecf97d975d759211d28e9a4d91.md) | 2,165 |  2,579,903 |  423 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e11f7868ca90c7ecf97d975d759211d28e9a4d91

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25791321284)
