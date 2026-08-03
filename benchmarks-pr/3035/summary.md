| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/fibonacci-814ff0ea541b5c900f1834ba90c40c59981c6467.md) | 483 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/keccak-814ff0ea541b5c900f1834ba90c40c59981c6467.md) | 7,392 |  14,365,133 |  1,532 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/sha2_bench-814ff0ea541b5c900f1834ba90c40c59981c6467.md) | 4,110 |  11,167,961 |  516 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/regex-814ff0ea541b5c900f1834ba90c40c59981c6467.md) | 658 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/ecrecover-814ff0ea541b5c900f1834ba90c40c59981c6467.md) | 252 |  78,475 |  225 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/pairing-814ff0ea541b5c900f1834ba90c40c59981c6467.md) | 236 |  592,827 |  196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/kitchen_sink-814ff0ea541b5c900f1834ba90c40c59981c6467.md) | 2,357 |  2,341,811 |  563 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/814ff0ea541b5c900f1834ba90c40c59981c6467

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30823904666)
