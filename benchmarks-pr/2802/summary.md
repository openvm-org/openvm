| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/fibonacci-d5e13530c975064c4027a26eb28bf5bf8ebf1f16.md) | 1,573 |  4,000,051 |  433 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/keccak-d5e13530c975064c4027a26eb28bf5bf8ebf1f16.md) | 13,771 |  14,365,133 |  2,184 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/sha2_bench-d5e13530c975064c4027a26eb28bf5bf8ebf1f16.md) | 9,047 |  11,167,961 |  1,401 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/regex-d5e13530c975064c4027a26eb28bf5bf8ebf1f16.md) | 1,462 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/ecrecover-d5e13530c975064c4027a26eb28bf5bf8ebf1f16.md) | 470 |  112,210 |  262 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/pairing-d5e13530c975064c4027a26eb28bf5bf8ebf1f16.md) | 594 |  592,827 |  253 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/kitchen_sink-d5e13530c975064c4027a26eb28bf5bf8ebf1f16.md) | 2,216 |  1,979,971 |  414 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d5e13530c975064c4027a26eb28bf5bf8ebf1f16

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26234943931)
