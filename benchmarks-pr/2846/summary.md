| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci-a3c4230087d6f7bcf23228398bca034cc58250da.md) | 4,039 |  12,000,265 | <span style='color: green'>(-3318 [-74.0%])</span> 1,168 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/keccak-a3c4230087d6f7bcf23228398bca034cc58250da.md) | 21,935 |  18,655,329 |  4,640 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/sha2_bench-a3c4230087d6f7bcf23228398bca034cc58250da.md) | 9,633 |  14,793,960 |  1,834 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex-a3c4230087d6f7bcf23228398bca034cc58250da.md) | 1,524 |  4,137,067 | <span style='color: green'>(-11563 [-96.4%])</span> 434 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover-a3c4230087d6f7bcf23228398bca034cc58250da.md) | 610 |  123,583 | <span style='color: green'>(-5567 [-95.1%])</span> 289 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing-a3c4230087d6f7bcf23228398bca034cc58250da.md) | 948 |  1,745,757 | <span style='color: green'>(-6080 [-95.3%])</span> 300 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink-a3c4230087d6f7bcf23228398bca034cc58250da.md) | 4,145 |  2,579,903 |  889 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/fibonacci_e2e-a3c4230087d6f7bcf23228398bca034cc58250da.md) | 1,712 |  12,000,265 |  495 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/regex_e2e-a3c4230087d6f7bcf23228398bca034cc58250da.md) | 721 |  4,137,067 |  200 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/ecrecover_e2e-a3c4230087d6f7bcf23228398bca034cc58250da.md) | 369 |  123,583 |  144 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/pairing_e2e-a3c4230087d6f7bcf23228398bca034cc58250da.md) | 505 |  1,745,757 |  146 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2846/kitchen_sink_e2e-a3c4230087d6f7bcf23228398bca034cc58250da.md) | 2,173 |  2,579,903 |  384 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a3c4230087d6f7bcf23228398bca034cc58250da

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27232922504)
