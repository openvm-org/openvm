| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2908/fibonacci-cf0db95b879fd993638b17ea73fe599b96b14821.md) | 1,397 |  4,000,051 |  394 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2908/keccak-cf0db95b879fd993638b17ea73fe599b96b14821.md) | 16,251 |  14,365,133 |  3,019 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2908/sha2_bench-cf0db95b879fd993638b17ea73fe599b96b14821.md) | 10,086 |  11,167,961 |  1,029 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2908/regex-cf0db95b879fd993638b17ea73fe599b96b14821.md) | 1,556 |  4,090,656 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2908/ecrecover-cf0db95b879fd993638b17ea73fe599b96b14821.md) | 442 |  112,210 |  309 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2908/pairing-cf0db95b879fd993638b17ea73fe599b96b14821.md) | 601 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2908/kitchen_sink-cf0db95b879fd993638b17ea73fe599b96b14821.md) | 3,894 |  1,979,971 |  856 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/cf0db95b879fd993638b17ea73fe599b96b14821

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27820182238)
