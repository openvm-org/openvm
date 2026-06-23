| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/fibonacci-f01cd17abdbf6b6d4933d3206f84a4bcd52dfd36.md) | 1,028 |  4,000,051 |  388 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/keccak-f01cd17abdbf6b6d4933d3206f84a4bcd52dfd36.md) | 15,676 |  14,365,133 |  3,002 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/sha2_bench-f01cd17abdbf6b6d4933d3206f84a4bcd52dfd36.md) | 8,081 |  11,167,961 |  988 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/regex-f01cd17abdbf6b6d4933d3206f84a4bcd52dfd36.md) | 1,206 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/ecrecover-f01cd17abdbf6b6d4933d3206f84a4bcd52dfd36.md) | 432 |  112,210 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/pairing-f01cd17abdbf6b6d4933d3206f84a4bcd52dfd36.md) | 599 |  592,827 |  298 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/kitchen_sink-f01cd17abdbf6b6d4933d3206f84a4bcd52dfd36.md) | 3,905 |  1,979,971 |  855 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f01cd17abdbf6b6d4933d3206f84a4bcd52dfd36

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28053566610)
