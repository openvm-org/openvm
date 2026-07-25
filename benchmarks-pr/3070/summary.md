| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/fibonacci-0544faba0b5d88e26e60e148fbc62dd2b18827fa.md) | 471 |  4,000,051 |  242 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/keccak-0544faba0b5d88e26e60e148fbc62dd2b18827fa.md) | 7,245 |  14,365,133 |  1,515 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/sha2_bench-0544faba0b5d88e26e60e148fbc62dd2b18827fa.md) | 4,748 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/regex-0544faba0b5d88e26e60e148fbc62dd2b18827fa.md) | 667 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/ecrecover-0544faba0b5d88e26e60e148fbc62dd2b18827fa.md) | 229 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/pairing-0544faba0b5d88e26e60e148fbc62dd2b18827fa.md) | 322 |  592,827 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/kitchen_sink-0544faba0b5d88e26e60e148fbc62dd2b18827fa.md) | 2,670 |  1,979,971 |  474 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0544faba0b5d88e26e60e148fbc62dd2b18827fa

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30151504135)
