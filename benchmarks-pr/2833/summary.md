| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/fibonacci-ace6d50ba49cadf70c8199e069ee67a9601b7cd9.md) | 3,756 |  12,000,265 |  919 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/keccak-ace6d50ba49cadf70c8199e069ee67a9601b7cd9.md) | 18,637 |  18,655,329 |  3,290 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/sha2_bench-ace6d50ba49cadf70c8199e069ee67a9601b7cd9.md) | 10,249 |  14,793,960 |  1,473 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/regex-ace6d50ba49cadf70c8199e069ee67a9601b7cd9.md) | 1,387 |  4,137,067 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/ecrecover-ace6d50ba49cadf70c8199e069ee67a9601b7cd9.md) | 608 |  123,583 |  256 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/pairing-ace6d50ba49cadf70c8199e069ee67a9601b7cd9.md) | 883 |  1,745,757 |  260 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/kitchen_sink-ace6d50ba49cadf70c8199e069ee67a9601b7cd9.md) | 1,897 |  2,579,903 |  410 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ace6d50ba49cadf70c8199e069ee67a9601b7cd9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26778201905)
