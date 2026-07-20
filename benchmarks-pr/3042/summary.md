| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/fibonacci-ea0b7e5141bcd33a8d9a8760a0b5df44a46d4415.md) | 411 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/keccak-ea0b7e5141bcd33a8d9a8760a0b5df44a46d4415.md) | 8,579 |  14,365,133 |  1,521 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/sha2_bench-ea0b7e5141bcd33a8d9a8760a0b5df44a46d4415.md) | 4,184 |  11,167,961 |  514 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/regex-ea0b7e5141bcd33a8d9a8760a0b5df44a46d4415.md) | 586 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/ecrecover-ea0b7e5141bcd33a8d9a8760a0b5df44a46d4415.md) | 220 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/pairing-ea0b7e5141bcd33a8d9a8760a0b5df44a46d4415.md) | 286 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3042/kitchen_sink-ea0b7e5141bcd33a8d9a8760a0b5df44a46d4415.md) | 1,930 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ea0b7e5141bcd33a8d9a8760a0b5df44a46d4415

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29760982624)
