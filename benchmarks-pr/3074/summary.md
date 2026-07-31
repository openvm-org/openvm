| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/fibonacci-33f964ca8583198e2989e947d4dede73b0bcbf98.md) | 481 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/keccak-33f964ca8583198e2989e947d4dede73b0bcbf98.md) | 7,406 |  14,365,133 |  1,534 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/sha2_bench-33f964ca8583198e2989e947d4dede73b0bcbf98.md) | 4,151 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/regex-33f964ca8583198e2989e947d4dede73b0bcbf98.md) | 643 |  4,090,656 |  211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/ecrecover-33f964ca8583198e2989e947d4dede73b0bcbf98.md) | 232 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/pairing-33f964ca8583198e2989e947d4dede73b0bcbf98.md) | 239 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/kitchen_sink-33f964ca8583198e2989e947d4dede73b0bcbf98.md) | 2,031 |  1,979,971 |  457 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/33f964ca8583198e2989e947d4dede73b0bcbf98

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30670420561)
