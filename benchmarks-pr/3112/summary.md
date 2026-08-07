| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/fibonacci-b052b9de112207a86595f7fe3b334a21fa0cdb63.md) | 465 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/keccak-b052b9de112207a86595f7fe3b334a21fa0cdb63.md) | 7,440 |  14,365,133 |  1,525 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/sha2_bench-b052b9de112207a86595f7fe3b334a21fa0cdb63.md) | 4,155 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/regex-b052b9de112207a86595f7fe3b334a21fa0cdb63.md) | 659 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/ecrecover-b052b9de112207a86595f7fe3b334a21fa0cdb63.md) | 197 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/pairing-b052b9de112207a86595f7fe3b334a21fa0cdb63.md) | 231 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3112/kitchen_sink-b052b9de112207a86595f7fe3b334a21fa0cdb63.md) | 2,026 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b052b9de112207a86595f7fe3b334a21fa0cdb63

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31205535649)
