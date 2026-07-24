| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/fibonacci-a47842ab3b6a2085faf3990a0240ae0b00139de4.md) | 473 |  4,000,051 |  239 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/keccak-a47842ab3b6a2085faf3990a0240ae0b00139de4.md) | 7,331 |  14,365,133 |  1,527 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/sha2_bench-a47842ab3b6a2085faf3990a0240ae0b00139de4.md) | 4,751 |  11,167,961 |  534 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/regex-a47842ab3b6a2085faf3990a0240ae0b00139de4.md) | 673 |  4,090,656 |  220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/ecrecover-a47842ab3b6a2085faf3990a0240ae0b00139de4.md) | 230 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/pairing-a47842ab3b6a2085faf3990a0240ae0b00139de4.md) | 316 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/kitchen_sink-a47842ab3b6a2085faf3990a0240ae0b00139de4.md) | 2,674 |  1,979,971 |  475 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a47842ab3b6a2085faf3990a0240ae0b00139de4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30128892693)
