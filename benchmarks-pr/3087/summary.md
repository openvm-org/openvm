| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/fibonacci-0c430c197a2e7c814b9eab328193f7a185d4f355.md) | 471 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/keccak-0c430c197a2e7c814b9eab328193f7a185d4f355.md) | 7,435 |  14,365,133 |  1,539 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/sha2_bench-0c430c197a2e7c814b9eab328193f7a185d4f355.md) | 4,166 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/regex-0c430c197a2e7c814b9eab328193f7a185d4f355.md) | 644 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/ecrecover-0c430c197a2e7c814b9eab328193f7a185d4f355.md) | 228 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/pairing-0c430c197a2e7c814b9eab328193f7a185d4f355.md) | 239 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/kitchen_sink-0c430c197a2e7c814b9eab328193f7a185d4f355.md) | 2,025 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0c430c197a2e7c814b9eab328193f7a185d4f355

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30806480402)
