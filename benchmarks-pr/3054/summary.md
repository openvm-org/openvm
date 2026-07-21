| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/fibonacci-522308bf00d33da26dbbc8efa37d9c7535b10b38.md) | 472 |  4,000,051 |  243 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/keccak-522308bf00d33da26dbbc8efa37d9c7535b10b38.md) | 7,318 |  14,365,133 |  1,535 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/sha2_bench-522308bf00d33da26dbbc8efa37d9c7535b10b38.md) | 4,698 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/regex-522308bf00d33da26dbbc8efa37d9c7535b10b38.md) | 672 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/ecrecover-522308bf00d33da26dbbc8efa37d9c7535b10b38.md) | 233 |  112,210 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/pairing-522308bf00d33da26dbbc8efa37d9c7535b10b38.md) | 321 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/kitchen_sink-522308bf00d33da26dbbc8efa37d9c7535b10b38.md) | 2,625 |  1,979,971 |  468 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/522308bf00d33da26dbbc8efa37d9c7535b10b38

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29860917772)
