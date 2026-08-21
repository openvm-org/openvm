| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/fibonacci-4d105031aced92e76436b9128a7f264213e4b00b.md) | 454 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/keccak-4d105031aced92e76436b9128a7f264213e4b00b.md) | 7,152 |  14,365,133 |  1,584 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/sha2_bench-4d105031aced92e76436b9128a7f264213e4b00b.md) | 4,034 |  11,167,961 |  514 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/regex-4d105031aced92e76436b9128a7f264213e4b00b.md) | 719 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/ecrecover-4d105031aced92e76436b9128a7f264213e4b00b.md) | 207 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/pairing-4d105031aced92e76436b9128a7f264213e4b00b.md) | 235 |  592,827 |  170 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/kitchen_sink-4d105031aced92e76436b9128a7f264213e4b00b.md) | 2,143 |  1,979,971 |  453 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4d105031aced92e76436b9128a7f264213e4b00b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32496982916)
