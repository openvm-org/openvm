| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/fibonacci-49bedd12f77df4d6cc9d75329508428a8e0dab84.md) | 410 |  4,000,051 |  236 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/keccak-49bedd12f77df4d6cc9d75329508428a8e0dab84.md) | 8,395 |  14,365,133 |  1,533 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/sha2_bench-49bedd12f77df4d6cc9d75329508428a8e0dab84.md) | 3,948 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/regex-49bedd12f77df4d6cc9d75329508428a8e0dab84.md) | 568 |  4,090,656 |  211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/ecrecover-49bedd12f77df4d6cc9d75329508428a8e0dab84.md) | 221 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/pairing-49bedd12f77df4d6cc9d75329508428a8e0dab84.md) | 281 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/kitchen_sink-49bedd12f77df4d6cc9d75329508428a8e0dab84.md) | 1,886 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/49bedd12f77df4d6cc9d75329508428a8e0dab84

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29452336097)
