| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/fibonacci-020479d7af67c20ed108c9e23ca117329e5c1775.md) | 1,025 |  4,000,051 |  388 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/keccak-020479d7af67c20ed108c9e23ca117329e5c1775.md) | 15,768 |  14,365,133 |  3,040 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/sha2_bench-020479d7af67c20ed108c9e23ca117329e5c1775.md) | 8,178 |  11,167,961 |  1,000 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/regex-020479d7af67c20ed108c9e23ca117329e5c1775.md) | 1,156 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/ecrecover-020479d7af67c20ed108c9e23ca117329e5c1775.md) | 431 |  112,210 |  280 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/pairing-020479d7af67c20ed108c9e23ca117329e5c1775.md) | 588 |  592,827 |  300 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/kitchen_sink-020479d7af67c20ed108c9e23ca117329e5c1775.md) | 3,919 |  1,979,971 |  870 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/020479d7af67c20ed108c9e23ca117329e5c1775

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28477643032)
