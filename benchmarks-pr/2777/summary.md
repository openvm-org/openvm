| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/fibonacci-e53664d30663bdf156e08a79eb6c350c5a501426.md) | 1,805 |  4,000,051 |  430 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/keccak-e53664d30663bdf156e08a79eb6c350c5a501426.md) | 14,204 |  14,365,133 |  2,407 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/sha2_bench-e53664d30663bdf156e08a79eb6c350c5a501426.md) | 8,275 |  11,167,961 |  908 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/regex-e53664d30663bdf156e08a79eb6c350c5a501426.md) | 1,563 |  4,090,656 |  361 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/ecrecover-e53664d30663bdf156e08a79eb6c350c5a501426.md) | 597 |  112,210 |  265 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/pairing-e53664d30663bdf156e08a79eb6c350c5a501426.md) | 750 |  592,827 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/kitchen_sink-e53664d30663bdf156e08a79eb6c350c5a501426.md) | 1,901 |  1,979,971 |  415 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e53664d30663bdf156e08a79eb6c350c5a501426

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25887697442)
