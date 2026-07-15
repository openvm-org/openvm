| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/fibonacci-260495890f29bcea0d164e15c25884495949c9a9.md) | 415 |  4,000,051 |  227 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/keccak-260495890f29bcea0d164e15c25884495949c9a9.md) | 8,560 |  14,365,133 |  1,544 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/sha2_bench-260495890f29bcea0d164e15c25884495949c9a9.md) | 4,135 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/regex-260495890f29bcea0d164e15c25884495949c9a9.md) | 505 |  4,090,656 |  193 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/ecrecover-260495890f29bcea0d164e15c25884495949c9a9.md) | 222 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/pairing-260495890f29bcea0d164e15c25884495949c9a9.md) | 263 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/kitchen_sink-260495890f29bcea0d164e15c25884495949c9a9.md) | 2,000 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/260495890f29bcea0d164e15c25884495949c9a9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29438087592)
