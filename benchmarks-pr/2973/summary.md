| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/fibonacci-64db95248528adafefd0c014a4a9c125875590df.md) | 412 |  4,000,051 |  225 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/keccak-64db95248528adafefd0c014a4a9c125875590df.md) | 8,378 |  14,365,133 |  1,524 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/sha2_bench-64db95248528adafefd0c014a4a9c125875590df.md) | 4,099 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/regex-64db95248528adafefd0c014a4a9c125875590df.md) | 490 |  4,090,656 |  189 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/ecrecover-64db95248528adafefd0c014a4a9c125875590df.md) | 222 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/pairing-64db95248528adafefd0c014a4a9c125875590df.md) | 271 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/kitchen_sink-64db95248528adafefd0c014a4a9c125875590df.md) | 1,986 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/64db95248528adafefd0c014a4a9c125875590df

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29424385970)
