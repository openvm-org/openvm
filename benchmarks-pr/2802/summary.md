| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/fibonacci-6193f443eba35b9dd3fb5c568aca63497adeb047.md) | 1,572 |  4,000,051 |  432 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/keccak-6193f443eba35b9dd3fb5c568aca63497adeb047.md) | 13,914 |  14,365,133 |  2,347 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/sha2_bench-6193f443eba35b9dd3fb5c568aca63497adeb047.md) | 9,307 |  11,167,961 |  1,434 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/regex-6193f443eba35b9dd3fb5c568aca63497adeb047.md) | 1,594 |  4,090,656 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/ecrecover-6193f443eba35b9dd3fb5c568aca63497adeb047.md) | 485 |  112,210 |  260 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/pairing-6193f443eba35b9dd3fb5c568aca63497adeb047.md) | 603 |  592,827 |  255 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/kitchen_sink-6193f443eba35b9dd3fb5c568aca63497adeb047.md) | 2,176 |  1,979,971 |  411 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6193f443eba35b9dd3fb5c568aca63497adeb047

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26880061171)
