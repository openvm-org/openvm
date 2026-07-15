| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/fibonacci-dde93584088d6960da8a293c1f96402f948c3e44.md) | 411 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/keccak-dde93584088d6960da8a293c1f96402f948c3e44.md) | 8,536 |  14,365,133 |  1,554 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/sha2_bench-dde93584088d6960da8a293c1f96402f948c3e44.md) | 3,957 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/regex-dde93584088d6960da8a293c1f96402f948c3e44.md) | 570 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/ecrecover-dde93584088d6960da8a293c1f96402f948c3e44.md) | 218 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/pairing-dde93584088d6960da8a293c1f96402f948c3e44.md) | 285 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2944/kitchen_sink-dde93584088d6960da8a293c1f96402f948c3e44.md) | 1,890 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/dde93584088d6960da8a293c1f96402f948c3e44

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29455141403)
