| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/fibonacci-a439e4cd487ece9cbe33ea07ff18694422172a23.md) | 470 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/keccak-a439e4cd487ece9cbe33ea07ff18694422172a23.md) | 8,741 |  14,365,133 |  1,521 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/sha2_bench-a439e4cd487ece9cbe33ea07ff18694422172a23.md) | 3,934 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/regex-a439e4cd487ece9cbe33ea07ff18694422172a23.md) | 499 |  4,090,656 |  192 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/ecrecover-a439e4cd487ece9cbe33ea07ff18694422172a23.md) | 219 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/pairing-a439e4cd487ece9cbe33ea07ff18694422172a23.md) | 272 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/kitchen_sink-a439e4cd487ece9cbe33ea07ff18694422172a23.md) | 1,916 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a439e4cd487ece9cbe33ea07ff18694422172a23

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29372893603)
