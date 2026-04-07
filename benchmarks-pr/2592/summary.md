| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-d13e237011e73decbea1c511be2bea6f11685461.md) | 3,813 |  12,000,265 |  949 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-d13e237011e73decbea1c511be2bea6f11685461.md) | 18,529 |  18,655,329 |  3,318 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-d13e237011e73decbea1c511be2bea6f11685461.md) | 1,419 |  4,137,067 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-d13e237011e73decbea1c511be2bea6f11685461.md) | 647 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-d13e237011e73decbea1c511be2bea6f11685461.md) | 913 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-d13e237011e73decbea1c511be2bea6f11685461.md) | 2,305 |  2,579,903 |  450 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d13e237011e73decbea1c511be2bea6f11685461

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24103853570)
