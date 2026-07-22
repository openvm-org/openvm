| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-7af0b55794e6ae12e8da770faa6ae41a56945b2b.md) | 480 |  4,000,051 |  240 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-7af0b55794e6ae12e8da770faa6ae41a56945b2b.md) | 7,325 |  14,365,133 |  1,538 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-7af0b55794e6ae12e8da770faa6ae41a56945b2b.md) | 4,709 |  11,167,961 |  528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-7af0b55794e6ae12e8da770faa6ae41a56945b2b.md) | 686 |  4,090,656 |  220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-7af0b55794e6ae12e8da770faa6ae41a56945b2b.md) | 228 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-7af0b55794e6ae12e8da770faa6ae41a56945b2b.md) | 277 |  592,827 |  191 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-7af0b55794e6ae12e8da770faa6ae41a56945b2b.md) | 2,733 |  1,979,971 |  468 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7af0b55794e6ae12e8da770faa6ae41a56945b2b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29962807295)
