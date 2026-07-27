| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3076/fibonacci-e8fe29b0dae1a97ee0f734094729ce85f4f7b45c.md) | 470 |  4,000,051 |  242 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3076/keccak-e8fe29b0dae1a97ee0f734094729ce85f4f7b45c.md) | 7,297 |  14,365,133 |  1,537 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3076/sha2_bench-e8fe29b0dae1a97ee0f734094729ce85f4f7b45c.md) | 4,750 |  11,167,961 |  528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3076/regex-e8fe29b0dae1a97ee0f734094729ce85f4f7b45c.md) | 667 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3076/ecrecover-e8fe29b0dae1a97ee0f734094729ce85f4f7b45c.md) | 228 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3076/pairing-e8fe29b0dae1a97ee0f734094729ce85f4f7b45c.md) | 320 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3076/kitchen_sink-e8fe29b0dae1a97ee0f734094729ce85f4f7b45c.md) | 2,673 |  1,979,971 |  474 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e8fe29b0dae1a97ee0f734094729ce85f4f7b45c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30308223757)
