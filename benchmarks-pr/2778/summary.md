| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-a79b34a054208c6c7e774630cf464565b9075296.md) | 1,638 |  4,000,051 |  540 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-a79b34a054208c6c7e774630cf464565b9075296.md) | 13,931 |  14,365,133 |  2,219 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-a79b34a054208c6c7e774630cf464565b9075296.md) | 9,499 |  11,167,961 |  1,435 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-a79b34a054208c6c7e774630cf464565b9075296.md) | 1,535 |  4,090,656 |  380 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-a79b34a054208c6c7e774630cf464565b9075296.md) | 513 |  112,210 |  289 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-a79b34a054208c6c7e774630cf464565b9075296.md) | 616 |  592,827 |  276 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-a79b34a054208c6c7e774630cf464565b9075296.md) | 1,962 |  1,979,971 |  431 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a79b34a054208c6c7e774630cf464565b9075296

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25830409979)
