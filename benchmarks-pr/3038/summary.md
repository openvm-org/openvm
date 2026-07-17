| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3038/fibonacci-1b65bfd4866b8e23b4f1c29abf11e85ee2a023f0.md) | 415 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3038/keccak-1b65bfd4866b8e23b4f1c29abf11e85ee2a023f0.md) | 8,698 |  14,365,133 |  1,545 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3038/sha2_bench-1b65bfd4866b8e23b4f1c29abf11e85ee2a023f0.md) | 4,215 |  11,167,961 |  532 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3038/regex-1b65bfd4866b8e23b4f1c29abf11e85ee2a023f0.md) | 561 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3038/ecrecover-1b65bfd4866b8e23b4f1c29abf11e85ee2a023f0.md) | 221 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3038/pairing-1b65bfd4866b8e23b4f1c29abf11e85ee2a023f0.md) | 281 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3038/kitchen_sink-1b65bfd4866b8e23b4f1c29abf11e85ee2a023f0.md) | 2,008 |  1,979,971 |  457 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1b65bfd4866b8e23b4f1c29abf11e85ee2a023f0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29617046411)
