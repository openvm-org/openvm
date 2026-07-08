| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/fibonacci-c63952f29dcc982f381412c4d38ee2fcb54c3acf.md) | 964 |  4,000,051 |  388 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/keccak-c63952f29dcc982f381412c4d38ee2fcb54c3acf.md) | 15,819 |  14,365,133 |  3,044 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/sha2_bench-c63952f29dcc982f381412c4d38ee2fcb54c3acf.md) | 8,136 |  11,167,961 |  1,002 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/regex-c63952f29dcc982f381412c4d38ee2fcb54c3acf.md) | 1,195 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/ecrecover-c63952f29dcc982f381412c4d38ee2fcb54c3acf.md) | 437 |  112,210 |  280 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/pairing-c63952f29dcc982f381412c4d38ee2fcb54c3acf.md) | 581 |  592,827 |  295 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2953/kitchen_sink-c63952f29dcc982f381412c4d38ee2fcb54c3acf.md) | 3,823 |  1,979,971 |  860 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c63952f29dcc982f381412c4d38ee2fcb54c3acf

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28953822141)
