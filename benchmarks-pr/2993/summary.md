| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/fibonacci-9efa857c85ed2ba4d2d50796c3a2c6dcc37869e3.md) | 407 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/keccak-9efa857c85ed2ba4d2d50796c3a2c6dcc37869e3.md) | 8,707 |  14,365,133 |  1,543 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/sha2_bench-9efa857c85ed2ba4d2d50796c3a2c6dcc37869e3.md) | 4,189 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/regex-9efa857c85ed2ba4d2d50796c3a2c6dcc37869e3.md) | 573 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/ecrecover-9efa857c85ed2ba4d2d50796c3a2c6dcc37869e3.md) | 221 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/pairing-9efa857c85ed2ba4d2d50796c3a2c6dcc37869e3.md) | 289 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/kitchen_sink-9efa857c85ed2ba4d2d50796c3a2c6dcc37869e3.md) | 1,935 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9efa857c85ed2ba4d2d50796c3a2c6dcc37869e3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29527730039)
