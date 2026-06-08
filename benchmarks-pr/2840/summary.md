| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/fibonacci-4c9cea5e937a64251c2716689f93f463132b5768.md) | 1,406 |  4,000,051 |  433 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/keccak-4c9cea5e937a64251c2716689f93f463132b5768.md) | 13,696 |  14,365,133 |  2,370 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/sha2_bench-4c9cea5e937a64251c2716689f93f463132b5768.md) | 8,824 |  11,167,961 |  1,392 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/regex-4c9cea5e937a64251c2716689f93f463132b5768.md) | 1,379 |  4,090,656 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/ecrecover-4c9cea5e937a64251c2716689f93f463132b5768.md) | 427 |  112,210 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/pairing-4c9cea5e937a64251c2716689f93f463132b5768.md) | 571 |  592,827 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/kitchen_sink-4c9cea5e937a64251c2716689f93f463132b5768.md) | 3,709 |  1,979,971 |  937 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4c9cea5e937a64251c2716689f93f463132b5768

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27145373959)
