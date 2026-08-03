| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3095/fibonacci-1d1e2c662cd1170fb50c532e9a9d1f1c528d41d5.md) | 474 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3095/keccak-1d1e2c662cd1170fb50c532e9a9d1f1c528d41d5.md) | 7,310 |  14,365,133 |  1,509 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3095/sha2_bench-1d1e2c662cd1170fb50c532e9a9d1f1c528d41d5.md) | 4,147 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3095/regex-1d1e2c662cd1170fb50c532e9a9d1f1c528d41d5.md) | 662 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3095/ecrecover-1d1e2c662cd1170fb50c532e9a9d1f1c528d41d5.md) | 231 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3095/pairing-1d1e2c662cd1170fb50c532e9a9d1f1c528d41d5.md) | 236 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3095/kitchen_sink-1d1e2c662cd1170fb50c532e9a9d1f1c528d41d5.md) | 2,042 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1d1e2c662cd1170fb50c532e9a9d1f1c528d41d5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30848183611)
