| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/fibonacci-c8d8d4cc5165e2a40f8bf586ecbba29dd27a26d4.md) | 471 |  4,000,051 |  243 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/keccak-c8d8d4cc5165e2a40f8bf586ecbba29dd27a26d4.md) | 7,335 |  14,365,133 |  1,544 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/sha2_bench-c8d8d4cc5165e2a40f8bf586ecbba29dd27a26d4.md) | 4,681 |  11,167,961 |  533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/regex-c8d8d4cc5165e2a40f8bf586ecbba29dd27a26d4.md) | 675 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/ecrecover-c8d8d4cc5165e2a40f8bf586ecbba29dd27a26d4.md) | 231 |  112,210 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/pairing-c8d8d4cc5165e2a40f8bf586ecbba29dd27a26d4.md) | 273 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/kitchen_sink-c8d8d4cc5165e2a40f8bf586ecbba29dd27a26d4.md) | 2,742 |  1,979,971 |  467 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c8d8d4cc5165e2a40f8bf586ecbba29dd27a26d4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30016157927)
