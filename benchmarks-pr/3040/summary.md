| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/fibonacci-dea1ecece00e7399a55912a292003449bdb26d9c.md) | 412 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/keccak-dea1ecece00e7399a55912a292003449bdb26d9c.md) | 8,580 |  14,365,133 |  1,520 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/sha2_bench-dea1ecece00e7399a55912a292003449bdb26d9c.md) | 4,201 |  11,167,961 |  518 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/regex-dea1ecece00e7399a55912a292003449bdb26d9c.md) | 575 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/ecrecover-dea1ecece00e7399a55912a292003449bdb26d9c.md) | 217 |  112,210 |  179 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/pairing-dea1ecece00e7399a55912a292003449bdb26d9c.md) | 299 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/kitchen_sink-dea1ecece00e7399a55912a292003449bdb26d9c.md) | 1,926 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/dea1ecece00e7399a55912a292003449bdb26d9c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29652392496)
