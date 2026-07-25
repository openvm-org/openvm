| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/fibonacci-b99914f7f06a56480959b2fd2f05ca57d5ea811c.md) | 464 |  4,000,051 |  241 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/keccak-b99914f7f06a56480959b2fd2f05ca57d5ea811c.md) | 7,316 |  14,365,133 |  1,542 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/sha2_bench-b99914f7f06a56480959b2fd2f05ca57d5ea811c.md) | 4,680 |  11,167,961 |  527 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/regex-b99914f7f06a56480959b2fd2f05ca57d5ea811c.md) | 669 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/ecrecover-b99914f7f06a56480959b2fd2f05ca57d5ea811c.md) | 227 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/pairing-b99914f7f06a56480959b2fd2f05ca57d5ea811c.md) | 321 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/kitchen_sink-b99914f7f06a56480959b2fd2f05ca57d5ea811c.md) | 2,669 |  1,979,971 |  472 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b99914f7f06a56480959b2fd2f05ca57d5ea811c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30140090984)
