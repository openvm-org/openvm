| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/fibonacci-569ae2557f2e51ea6ae01d4f307624a8441c4605.md) | 1,039 |  4,000,051 |  396 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/keccak-569ae2557f2e51ea6ae01d4f307624a8441c4605.md) | 15,856 |  14,365,133 |  3,048 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/sha2_bench-569ae2557f2e51ea6ae01d4f307624a8441c4605.md) | 8,286 |  11,167,961 |  1,018 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/regex-569ae2557f2e51ea6ae01d4f307624a8441c4605.md) | 1,173 |  4,090,656 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/ecrecover-569ae2557f2e51ea6ae01d4f307624a8441c4605.md) | 428 |  112,210 |  285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/pairing-569ae2557f2e51ea6ae01d4f307624a8441c4605.md) | 591 |  592,827 |  297 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/kitchen_sink-569ae2557f2e51ea6ae01d4f307624a8441c4605.md) | 3,852 |  1,979,971 |  856 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/569ae2557f2e51ea6ae01d4f307624a8441c4605

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28247509623)
