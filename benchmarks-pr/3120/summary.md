| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/fibonacci-5669f1a94cc87db8e3ed34eb68743b8fc56691a7.md) | 490 |  4,000,051 |  237 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/keccak-5669f1a94cc87db8e3ed34eb68743b8fc56691a7.md) | 7,723 |  14,365,133 |  1,630 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/sha2_bench-5669f1a94cc87db8e3ed34eb68743b8fc56691a7.md) | 4,350 |  11,167,961 |  530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/regex-5669f1a94cc87db8e3ed34eb68743b8fc56691a7.md) | 769 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/ecrecover-5669f1a94cc87db8e3ed34eb68743b8fc56691a7.md) | 212 |  112,210 |  192 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/pairing-5669f1a94cc87db8e3ed34eb68743b8fc56691a7.md) | 249 |  592,827 |  171 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/kitchen_sink-5669f1a94cc87db8e3ed34eb68743b8fc56691a7.md) | 2,233 |  1,979,971 |  469 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5669f1a94cc87db8e3ed34eb68743b8fc56691a7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33171440178)
