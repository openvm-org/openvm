| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/fibonacci-05f0b327759e3b3218ef8a750cbc3abc218f0955.md) | 3,756 |  12,000,265 |  919 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/keccak-05f0b327759e3b3218ef8a750cbc3abc218f0955.md) | 18,539 |  18,655,329 |  3,275 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/sha2_bench-05f0b327759e3b3218ef8a750cbc3abc218f0955.md) | 10,030 |  14,793,960 |  1,442 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/regex-05f0b327759e3b3218ef8a750cbc3abc218f0955.md) | 1,387 |  4,137,067 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/ecrecover-05f0b327759e3b3218ef8a750cbc3abc218f0955.md) | 595 |  123,583 |  251 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/pairing-05f0b327759e3b3218ef8a750cbc3abc218f0955.md) | 891 |  1,745,757 |  264 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2800/kitchen_sink-05f0b327759e3b3218ef8a750cbc3abc218f0955.md) | 1,892 |  2,579,903 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/05f0b327759e3b3218ef8a750cbc3abc218f0955

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26254052570)
