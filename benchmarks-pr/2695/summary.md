| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/fibonacci-adffbe7e5e9f407eb022ca6e687e5cbf1f6a7685.md) | 3,825 |  12,000,265 |  958 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/keccak-adffbe7e5e9f407eb022ca6e687e5cbf1f6a7685.md) | 18,546 |  18,655,329 |  3,293 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/sha2_bench-adffbe7e5e9f407eb022ca6e687e5cbf1f6a7685.md) | 8,897 |  14,793,960 |  1,383 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/regex-adffbe7e5e9f407eb022ca6e687e5cbf1f6a7685.md) | 1,409 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/ecrecover-adffbe7e5e9f407eb022ca6e687e5cbf1f6a7685.md) | 642 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/pairing-adffbe7e5e9f407eb022ca6e687e5cbf1f6a7685.md) | 910 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/kitchen_sink-adffbe7e5e9f407eb022ca6e687e5cbf1f6a7685.md) | 2,103 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/adffbe7e5e9f407eb022ca6e687e5cbf1f6a7685

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24419092314)
