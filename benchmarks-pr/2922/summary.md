| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/fibonacci-65899a324410852ccee8b8c4feaa05fa1525e638.md) | 465 |  4,000,051 |  226 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/keccak-65899a324410852ccee8b8c4feaa05fa1525e638.md) | 8,748 |  14,365,133 |  1,514 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/sha2_bench-65899a324410852ccee8b8c4feaa05fa1525e638.md) | 3,920 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/regex-65899a324410852ccee8b8c4feaa05fa1525e638.md) | 499 |  4,090,656 |  190 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/ecrecover-65899a324410852ccee8b8c4feaa05fa1525e638.md) | 220 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/pairing-65899a324410852ccee8b8c4feaa05fa1525e638.md) | 275 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/kitchen_sink-65899a324410852ccee8b8c4feaa05fa1525e638.md) | 1,903 |  1,979,971 |  456 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/65899a324410852ccee8b8c4feaa05fa1525e638

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29360603247)
