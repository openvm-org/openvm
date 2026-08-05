| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-c6b41bc0a462e23b64e1aac74b9ede42f38d7807.md) | 899 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-c6b41bc0a462e23b64e1aac74b9ede42f38d7807.md) | 8,580 |  14,365,133 |  1,517 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-c6b41bc0a462e23b64e1aac74b9ede42f38d7807.md) | 4,209 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-c6b41bc0a462e23b64e1aac74b9ede42f38d7807.md) | 734 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-c6b41bc0a462e23b64e1aac74b9ede42f38d7807.md) | 496 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-c6b41bc0a462e23b64e1aac74b9ede42f38d7807.md) | 475 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-c6b41bc0a462e23b64e1aac74b9ede42f38d7807.md) | 2,338 |  1,979,971 |  453 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c6b41bc0a462e23b64e1aac74b9ede42f38d7807

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31027301647)
