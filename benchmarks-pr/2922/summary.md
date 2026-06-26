| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/fibonacci-26202ea58256d7a9872c15b9dc090827fa68fb7b.md) | 1,033 |  4,000,051 |  390 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/keccak-26202ea58256d7a9872c15b9dc090827fa68fb7b.md) | 15,530 |  14,365,133 |  3,025 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/sha2_bench-26202ea58256d7a9872c15b9dc090827fa68fb7b.md) | 7,831 |  11,167,961 |  993 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/regex-26202ea58256d7a9872c15b9dc090827fa68fb7b.md) | 1,024 |  4,090,656 |  300 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/ecrecover-26202ea58256d7a9872c15b9dc090827fa68fb7b.md) | 426 |  112,210 |  280 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/pairing-26202ea58256d7a9872c15b9dc090827fa68fb7b.md) | 561 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/kitchen_sink-26202ea58256d7a9872c15b9dc090827fa68fb7b.md) | 3,744 |  1,979,971 |  848 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/26202ea58256d7a9872c15b9dc090827fa68fb7b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28259606771)
