| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/fibonacci-5dfb2b8c3417d4bf22c1599d4a940cc561b5a1eb.md) | 456 |  4,000,051 |  240 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/keccak-5dfb2b8c3417d4bf22c1599d4a940cc561b5a1eb.md) | 7,294 |  14,365,133 |  1,539 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/sha2_bench-5dfb2b8c3417d4bf22c1599d4a940cc561b5a1eb.md) | 4,715 |  11,167,961 |  538 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/regex-5dfb2b8c3417d4bf22c1599d4a940cc561b5a1eb.md) | 647 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/ecrecover-5dfb2b8c3417d4bf22c1599d4a940cc561b5a1eb.md) | 227 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/pairing-5dfb2b8c3417d4bf22c1599d4a940cc561b5a1eb.md) | 304 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/kitchen_sink-5dfb2b8c3417d4bf22c1599d4a940cc561b5a1eb.md) | 2,649 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5dfb2b8c3417d4bf22c1599d4a940cc561b5a1eb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30487108907)
