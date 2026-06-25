| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/fibonacci-85f743401cfc5874323922cae3dc026e8318d45a.md) | 1,020 |  4,000,051 |  389 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/keccak-85f743401cfc5874323922cae3dc026e8318d45a.md) | 15,803 |  14,365,133 |  3,040 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/sha2_bench-85f743401cfc5874323922cae3dc026e8318d45a.md) | 8,291 |  11,167,961 |  1,010 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/regex-85f743401cfc5874323922cae3dc026e8318d45a.md) | 1,187 |  4,090,656 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/ecrecover-85f743401cfc5874323922cae3dc026e8318d45a.md) | 434 |  112,210 |  277 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/pairing-85f743401cfc5874323922cae3dc026e8318d45a.md) | 601 |  592,827 |  296 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/kitchen_sink-85f743401cfc5874323922cae3dc026e8318d45a.md) | 3,923 |  1,979,971 |  867 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/85f743401cfc5874323922cae3dc026e8318d45a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28192366344)
