| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/fibonacci-30838852793d6a557c5df2157699653403128821.md) | 476 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/keccak-30838852793d6a557c5df2157699653403128821.md) | 7,630 |  14,365,133 |  1,627 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/sha2_bench-30838852793d6a557c5df2157699653403128821.md) | 4,341 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/regex-30838852793d6a557c5df2157699653403128821.md) | 765 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/ecrecover-30838852793d6a557c5df2157699653403128821.md) | 209 |  112,210 |  189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/pairing-30838852793d6a557c5df2157699653403128821.md) | 251 |  592,827 |  173 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/kitchen_sink-30838852793d6a557c5df2157699653403128821.md) | 2,232 |  1,979,971 |  473 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/30838852793d6a557c5df2157699653403128821

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33188633577)
