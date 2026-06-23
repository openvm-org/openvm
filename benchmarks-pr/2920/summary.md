| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/fibonacci-8fa2c50c1963680628e40dadf029542bce5d0b65.md) | 1,033 |  4,000,051 |  392 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/keccak-8fa2c50c1963680628e40dadf029542bce5d0b65.md) | 16,424 |  14,365,133 |  3,046 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/sha2_bench-8fa2c50c1963680628e40dadf029542bce5d0b65.md) | 8,089 |  11,167,961 |  990 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/regex-8fa2c50c1963680628e40dadf029542bce5d0b65.md) | 1,231 |  4,090,656 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/ecrecover-8fa2c50c1963680628e40dadf029542bce5d0b65.md) | 437 |  112,210 |  279 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/pairing-8fa2c50c1963680628e40dadf029542bce5d0b65.md) | 595 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2920/kitchen_sink-8fa2c50c1963680628e40dadf029542bce5d0b65.md) | 3,859 |  1,979,971 |  854 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8fa2c50c1963680628e40dadf029542bce5d0b65

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28057477256)
