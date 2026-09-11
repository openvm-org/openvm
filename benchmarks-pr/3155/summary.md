| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/fibonacci-c1340d31e0b3ef68f1c34b67d584f5fb7d45b156.md) | 482 |  4,000,051 |  238 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/keccak-c1340d31e0b3ef68f1c34b67d584f5fb7d45b156.md) | 7,439 |  14,365,133 |  1,591 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/sha2_bench-c1340d31e0b3ef68f1c34b67d584f5fb7d45b156.md) | 4,332 |  11,167,961 |  527 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/regex-c1340d31e0b3ef68f1c34b67d584f5fb7d45b156.md) | 769 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/ecrecover-c1340d31e0b3ef68f1c34b67d584f5fb7d45b156.md) | 208 |  112,210 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/pairing-c1340d31e0b3ef68f1c34b67d584f5fb7d45b156.md) | 251 |  592,827 |  173 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/kitchen_sink-c1340d31e0b3ef68f1c34b67d584f5fb7d45b156.md) | 2,231 |  1,979,971 |  472 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c1340d31e0b3ef68f1c34b67d584f5fb7d45b156

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/34606080450)
