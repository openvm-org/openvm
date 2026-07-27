| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/fibonacci-07dd5222a4a4ea8c062746227124c742ee5e7f10.md) | 499 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/keccak-07dd5222a4a4ea8c062746227124c742ee5e7f10.md) | 7,443 |  14,365,133 |  1,526 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/sha2_bench-07dd5222a4a4ea8c062746227124c742ee5e7f10.md) | 4,165 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/regex-07dd5222a4a4ea8c062746227124c742ee5e7f10.md) | 681 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/ecrecover-07dd5222a4a4ea8c062746227124c742ee5e7f10.md) | 231 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/pairing-07dd5222a4a4ea8c062746227124c742ee5e7f10.md) | 236 |  592,827 |  181 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/kitchen_sink-07dd5222a4a4ea8c062746227124c742ee5e7f10.md) | 2,059 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/07dd5222a4a4ea8c062746227124c742ee5e7f10

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30311179043)
