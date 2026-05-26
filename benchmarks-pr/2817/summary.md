| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/fibonacci-9537aa9e2c4933bdf4a577d769cf5a89fd43af71.md) | 3,747 |  12,000,265 |  915 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/keccak-9537aa9e2c4933bdf4a577d769cf5a89fd43af71.md) | 18,716 |  18,655,329 |  3,303 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/sha2_bench-9537aa9e2c4933bdf4a577d769cf5a89fd43af71.md) | 10,051 |  14,793,960 |  1,443 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/regex-9537aa9e2c4933bdf4a577d769cf5a89fd43af71.md) | 1,403 |  4,137,067 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/ecrecover-9537aa9e2c4933bdf4a577d769cf5a89fd43af71.md) | 594 |  123,583 |  249 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/pairing-9537aa9e2c4933bdf4a577d769cf5a89fd43af71.md) | 890 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/kitchen_sink-9537aa9e2c4933bdf4a577d769cf5a89fd43af71.md) | 1,908 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9537aa9e2c4933bdf4a577d769cf5a89fd43af71

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26472586966)
