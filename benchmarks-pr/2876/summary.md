| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/fibonacci-e4a4ec4f76c8b84530f198252a8e7a57f7c9d10f.md) | 1,685 |  4,000,051 |  540 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/keccak-e4a4ec4f76c8b84530f198252a8e7a57f7c9d10f.md) | 16,393 |  14,365,133 |  3,039 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/sha2_bench-e4a4ec4f76c8b84530f198252a8e7a57f7c9d10f.md) | 10,379 |  11,167,961 |  1,930 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/regex-e4a4ec4f76c8b84530f198252a8e7a57f7c9d10f.md) | 1,529 |  4,090,656 |  433 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/ecrecover-e4a4ec4f76c8b84530f198252a8e7a57f7c9d10f.md) | 491 |  112,210 |  310 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/pairing-e4a4ec4f76c8b84530f198252a8e7a57f7c9d10f.md) | 619 |  592,827 |  292 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2876/kitchen_sink-e4a4ec4f76c8b84530f198252a8e7a57f7c9d10f.md) | 3,912 |  1,979,971 |  852 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e4a4ec4f76c8b84530f198252a8e7a57f7c9d10f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27352846289)
