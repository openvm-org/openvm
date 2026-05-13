| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-0958ffc0da5eb187bb192709d23b71e61bd78e2c.md) | 1,899 |  4,000,051 |  537 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-0958ffc0da5eb187bb192709d23b71e61bd78e2c.md) | 13,741 |  14,365,133 |  2,265 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-0958ffc0da5eb187bb192709d23b71e61bd78e2c.md) | 9,523 |  11,167,961 |  1,422 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-0958ffc0da5eb187bb192709d23b71e61bd78e2c.md) | 1,625 |  4,090,656 |  382 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-0958ffc0da5eb187bb192709d23b71e61bd78e2c.md) | 641 |  112,210 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-0958ffc0da5eb187bb192709d23b71e61bd78e2c.md) | 760 |  592,827 |  275 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-0958ffc0da5eb187bb192709d23b71e61bd78e2c.md) | 2,033 |  1,979,971 |  430 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0958ffc0da5eb187bb192709d23b71e61bd78e2c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25806635210)
