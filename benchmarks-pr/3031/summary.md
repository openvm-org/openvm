| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3031/fibonacci-907c957124b0fe81daf9e519d190fe6148023d0d.md) | 406 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3031/keccak-907c957124b0fe81daf9e519d190fe6148023d0d.md) | 8,554 |  14,365,133 |  1,516 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3031/sha2_bench-907c957124b0fe81daf9e519d190fe6148023d0d.md) | 4,195 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3031/regex-907c957124b0fe81daf9e519d190fe6148023d0d.md) | 573 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3031/ecrecover-907c957124b0fe81daf9e519d190fe6148023d0d.md) | 217 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3031/pairing-907c957124b0fe81daf9e519d190fe6148023d0d.md) | 292 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3031/kitchen_sink-907c957124b0fe81daf9e519d190fe6148023d0d.md) | 1,917 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/907c957124b0fe81daf9e519d190fe6148023d0d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29572899013)
