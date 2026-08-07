| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-761dcb306db7d19e8526043703154f46a1e4d855.md) | 442 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-761dcb306db7d19e8526043703154f46a1e4d855.md) | 7,157 |  14,365,133 |  1,600 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-761dcb306db7d19e8526043703154f46a1e4d855.md) | 4,118 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-761dcb306db7d19e8526043703154f46a1e4d855.md) | 714 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-761dcb306db7d19e8526043703154f46a1e4d855.md) | 207 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-761dcb306db7d19e8526043703154f46a1e4d855.md) | 238 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-761dcb306db7d19e8526043703154f46a1e4d855.md) | 2,168 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/761dcb306db7d19e8526043703154f46a1e4d855

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31205559835)
