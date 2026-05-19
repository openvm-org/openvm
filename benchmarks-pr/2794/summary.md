| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/fibonacci-d30973fd5a2fa9b6a654e52b15321c3d868021f9.md) | 1,578 |  4,000,051 |  436 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/keccak-d30973fd5a2fa9b6a654e52b15321c3d868021f9.md) | 14,040 |  14,365,133 |  2,394 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/sha2_bench-d30973fd5a2fa9b6a654e52b15321c3d868021f9.md) | 9,363 |  11,167,961 |  1,449 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/regex-d30973fd5a2fa9b6a654e52b15321c3d868021f9.md) | 1,478 |  4,090,656 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/ecrecover-d30973fd5a2fa9b6a654e52b15321c3d868021f9.md) | 476 |  112,210 |  258 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/pairing-d30973fd5a2fa9b6a654e52b15321c3d868021f9.md) | 598 |  592,827 |  260 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/kitchen_sink-d30973fd5a2fa9b6a654e52b15321c3d868021f9.md) | 1,823 |  1,979,971 |  407 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d30973fd5a2fa9b6a654e52b15321c3d868021f9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26124752517)
