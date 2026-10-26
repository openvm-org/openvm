| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-77bb52e209a0219911199d8422149cec2c493541.md) | 1,686 |  12,000,265 |  367 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-77bb52e209a0219911199d8422149cec2c493541.md) | 9,674 |  18,655,329 |  1,555 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-77bb52e209a0219911199d8422149cec2c493541.md) | 5,340 |  14,793,960 |  592 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-77bb52e209a0219911199d8422149cec2c493541.md) | 704 |  4,137,067 |  220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-77bb52e209a0219911199d8422149cec2c493541.md) | 435 |  123,583 |  189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-77bb52e209a0219911199d8422149cec2c493541.md) | 606 |  1,745,757 |  194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-77bb52e209a0219911199d8422149cec2c493541.md) | 2,288 |  2,579,903 |  494 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/77bb52e209a0219911199d8422149cec2c493541

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/40391405607)
