| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/fibonacci-31bc031910f3268a20b7af3bf75b7c1cdad9b046.md) | 3,789 |  12,000,265 |  943 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/keccak-31bc031910f3268a20b7af3bf75b7c1cdad9b046.md) | 18,736 |  18,655,329 |  3,343 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/sha2_bench-31bc031910f3268a20b7af3bf75b7c1cdad9b046.md) | 9,002 |  14,793,960 |  1,397 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/regex-31bc031910f3268a20b7af3bf75b7c1cdad9b046.md) | 1,399 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/ecrecover-31bc031910f3268a20b7af3bf75b7c1cdad9b046.md) | 633 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/pairing-31bc031910f3268a20b7af3bf75b7c1cdad9b046.md) | 894 |  1,745,757 |  287 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/kitchen_sink-31bc031910f3268a20b7af3bf75b7c1cdad9b046.md) | 2,096 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/31bc031910f3268a20b7af3bf75b7c1cdad9b046

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25204841856)
