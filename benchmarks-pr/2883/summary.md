| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/fibonacci-431bff9fe725a5aec2d6d4e2fa1b7f379f4b8bab.md) | 1,358 |  4,000,051 |  519 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/keccak-431bff9fe725a5aec2d6d4e2fa1b7f379f4b8bab.md) | 16,319 |  14,365,133 |  3,003 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/sha2_bench-431bff9fe725a5aec2d6d4e2fa1b7f379f4b8bab.md) | 8,785 |  11,167,961 |  1,148 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/regex-431bff9fe725a5aec2d6d4e2fa1b7f379f4b8bab.md) | 1,471 |  4,090,656 |  426 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/ecrecover-431bff9fe725a5aec2d6d4e2fa1b7f379f4b8bab.md) | 477 |  112,210 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/pairing-431bff9fe725a5aec2d6d4e2fa1b7f379f4b8bab.md) | 616 |  592,827 |  291 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2883/kitchen_sink-431bff9fe725a5aec2d6d4e2fa1b7f379f4b8bab.md) | 3,923 |  1,979,971 |  863 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/431bff9fe725a5aec2d6d4e2fa1b7f379f4b8bab

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27571351819)
