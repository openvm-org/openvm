| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2805/fibonacci-53d1b4b48927645292bdfbbb097dd003f8abfc3c.md) | 1,549 |  4,000,051 |  432 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2805/keccak-53d1b4b48927645292bdfbbb097dd003f8abfc3c.md) | 13,796 |  14,365,133 |  2,368 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2805/sha2_bench-53d1b4b48927645292bdfbbb097dd003f8abfc3c.md) | 9,217 |  11,167,961 |  1,396 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2805/regex-53d1b4b48927645292bdfbbb097dd003f8abfc3c.md) | 1,460 |  4,090,656 |  364 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2805/ecrecover-53d1b4b48927645292bdfbbb097dd003f8abfc3c.md) | 471 |  112,210 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2805/pairing-53d1b4b48927645292bdfbbb097dd003f8abfc3c.md) | 592 |  592,827 |  255 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2805/kitchen_sink-53d1b4b48927645292bdfbbb097dd003f8abfc3c.md) | 2,140 |  1,979,971 |  407 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/53d1b4b48927645292bdfbbb097dd003f8abfc3c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26253796474)
