| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/fibonacci-f155f0e36f505fb9ed2ac1ec4c34de77e3598228.md) | 480 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/keccak-f155f0e36f505fb9ed2ac1ec4c34de77e3598228.md) | 7,436 |  14,365,133 |  1,546 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/sha2_bench-f155f0e36f505fb9ed2ac1ec4c34de77e3598228.md) | 4,156 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/regex-f155f0e36f505fb9ed2ac1ec4c34de77e3598228.md) | 660 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/ecrecover-f155f0e36f505fb9ed2ac1ec4c34de77e3598228.md) | 223 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/pairing-f155f0e36f505fb9ed2ac1ec4c34de77e3598228.md) | 235 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/kitchen_sink-f155f0e36f505fb9ed2ac1ec4c34de77e3598228.md) | 2,030 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f155f0e36f505fb9ed2ac1ec4c34de77e3598228

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30948792267)
