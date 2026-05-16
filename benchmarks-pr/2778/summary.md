| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-04d5961665bb43507373cf0fd67b070c28b24074.md) | 1,404 |  4,000,051 |  436 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-04d5961665bb43507373cf0fd67b070c28b24074.md) | 13,463 |  14,365,133 |  2,229 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-04d5961665bb43507373cf0fd67b070c28b24074.md) | 8,944 |  11,167,961 |  1,405 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-04d5961665bb43507373cf0fd67b070c28b24074.md) | 1,351 |  4,090,656 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-04d5961665bb43507373cf0fd67b070c28b24074.md) | 466 |  112,210 |  265 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-04d5961665bb43507373cf0fd67b070c28b24074.md) | 590 |  592,827 |  252 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-04d5961665bb43507373cf0fd67b070c28b24074.md) | 2,197 |  1,979,971 |  408 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/04d5961665bb43507373cf0fd67b070c28b24074

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25969619971)
