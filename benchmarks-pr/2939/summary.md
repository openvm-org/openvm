| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/fibonacci-ba2322e54aa939b1dd68326fa36df973ca49f4c7.md) | 1,029 |  4,000,051 |  388 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/keccak-ba2322e54aa939b1dd68326fa36df973ca49f4c7.md) | 15,796 |  14,365,133 |  3,030 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/sha2_bench-ba2322e54aa939b1dd68326fa36df973ca49f4c7.md) | 8,123 |  11,167,961 |  995 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/regex-ba2322e54aa939b1dd68326fa36df973ca49f4c7.md) | 1,161 |  4,090,656 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/ecrecover-ba2322e54aa939b1dd68326fa36df973ca49f4c7.md) | 438 |  112,210 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/pairing-ba2322e54aa939b1dd68326fa36df973ca49f4c7.md) | 602 |  592,827 |  299 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/kitchen_sink-ba2322e54aa939b1dd68326fa36df973ca49f4c7.md) | 3,903 |  1,979,971 |  874 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ba2322e54aa939b1dd68326fa36df973ca49f4c7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28389516590)
