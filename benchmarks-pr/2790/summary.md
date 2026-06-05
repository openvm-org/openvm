| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci-2d21d6303559c989dc3f71617a725476f1ef328d.md) | 3,691 |  12,000,265 |  909 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/keccak-2d21d6303559c989dc3f71617a725476f1ef328d.md) | 17,918 |  18,655,329 |  3,253 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/sha2_bench-2d21d6303559c989dc3f71617a725476f1ef328d.md) | 10,056 |  14,793,960 |  1,456 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex-2d21d6303559c989dc3f71617a725476f1ef328d.md) | 1,421 |  4,137,067 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover-2d21d6303559c989dc3f71617a725476f1ef328d.md) | 597 |  123,583 |  247 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing-2d21d6303559c989dc3f71617a725476f1ef328d.md) | 883 |  1,745,757 |  257 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink-2d21d6303559c989dc3f71617a725476f1ef328d.md) | 3,821 |  2,579,903 |  944 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/fibonacci_e2e-2d21d6303559c989dc3f71617a725476f1ef328d.md) | 1,628 |  12,000,265 |  414 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/regex_e2e-2d21d6303559c989dc3f71617a725476f1ef328d.md) | 665 |  4,137,067 |  169 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/ecrecover_e2e-2d21d6303559c989dc3f71617a725476f1ef328d.md) | 361 |  123,583 |  130 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/pairing_e2e-2d21d6303559c989dc3f71617a725476f1ef328d.md) | 482 |  1,745,757 |  130 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2790/kitchen_sink_e2e-2d21d6303559c989dc3f71617a725476f1ef328d.md) | 1,817 |  2,579,903 |  397 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2d21d6303559c989dc3f71617a725476f1ef328d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27014898523)
