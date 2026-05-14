| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/fibonacci-4ebbd3f2ee57e31e7b8c7653c02168c37a4b258d.md) | 1,838 |  4,000,051 |  451 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/keccak-4ebbd3f2ee57e31e7b8c7653c02168c37a4b258d.md) | 13,986 |  14,365,133 |  2,221 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/sha2_bench-4ebbd3f2ee57e31e7b8c7653c02168c37a4b258d.md) | 8,476 |  11,167,961 |  931 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/regex-4ebbd3f2ee57e31e7b8c7653c02168c37a4b258d.md) | 1,604 |  4,090,656 |  382 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/ecrecover-4ebbd3f2ee57e31e7b8c7653c02168c37a4b258d.md) | 638 |  112,210 |  289 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/pairing-4ebbd3f2ee57e31e7b8c7653c02168c37a4b258d.md) | 755 |  592,827 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/kitchen_sink-4ebbd3f2ee57e31e7b8c7653c02168c37a4b258d.md) | 2,019 |  1,979,971 |  428 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4ebbd3f2ee57e31e7b8c7653c02168c37a4b258d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25886111695)
