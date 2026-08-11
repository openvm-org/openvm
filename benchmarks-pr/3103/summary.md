| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-1f52593144dd17dcd16beae7ed35d4497a73bf54.md) | 470 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-1f52593144dd17dcd16beae7ed35d4497a73bf54.md) | 7,334 |  14,365,133 |  1,517 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-1f52593144dd17dcd16beae7ed35d4497a73bf54.md) | 4,136 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-1f52593144dd17dcd16beae7ed35d4497a73bf54.md) | 658 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-1f52593144dd17dcd16beae7ed35d4497a73bf54.md) | 222 |  112,210 |  194 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-1f52593144dd17dcd16beae7ed35d4497a73bf54.md) | 231 |  592,827 |  198 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-1f52593144dd17dcd16beae7ed35d4497a73bf54.md) | 2,030 |  1,979,971 |  529 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1f52593144dd17dcd16beae7ed35d4497a73bf54

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31508610245)
