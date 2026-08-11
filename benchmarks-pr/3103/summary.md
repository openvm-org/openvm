| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-a56fa0664d5ecbab92bd5c4849de05dfae15fd3d.md) | 474 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-a56fa0664d5ecbab92bd5c4849de05dfae15fd3d.md) | 7,318 |  14,365,133 |  1,525 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-a56fa0664d5ecbab92bd5c4849de05dfae15fd3d.md) | 4,127 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-a56fa0664d5ecbab92bd5c4849de05dfae15fd3d.md) | 654 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-a56fa0664d5ecbab92bd5c4849de05dfae15fd3d.md) | 223 |  112,210 |  194 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-a56fa0664d5ecbab92bd5c4849de05dfae15fd3d.md) | 231 |  592,827 |  197 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-a56fa0664d5ecbab92bd5c4849de05dfae15fd3d.md) | 2,048 |  1,979,971 |  538 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a56fa0664d5ecbab92bd5c4849de05dfae15fd3d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31527772426)
