| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-bf6e25f75dd5e702e27dffb125b84c549df5957d.md) | 1,904 |  4,000,051 |  527 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-bf6e25f75dd5e702e27dffb125b84c549df5957d.md) | 13,449 |  14,365,133 |  2,193 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-bf6e25f75dd5e702e27dffb125b84c549df5957d.md) | 9,473 |  11,167,961 |  1,416 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-bf6e25f75dd5e702e27dffb125b84c549df5957d.md) | 1,587 |  4,090,656 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-bf6e25f75dd5e702e27dffb125b84c549df5957d.md) | 598 |  112,210 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-bf6e25f75dd5e702e27dffb125b84c549df5957d.md) | 735 |  592,827 |  260 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-bf6e25f75dd5e702e27dffb125b84c549df5957d.md) | 1,868 |  1,979,971 |  405 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bf6e25f75dd5e702e27dffb125b84c549df5957d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25938244808)
