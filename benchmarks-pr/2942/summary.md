| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/fibonacci-c7a9ad2ee0e91dd5e13aa26f22a74287abcacbc0.md) | 864 |  4,000,051 |  396 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/keccak-c7a9ad2ee0e91dd5e13aa26f22a74287abcacbc0.md) | 15,606 |  14,365,133 |  3,076 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/sha2_bench-c7a9ad2ee0e91dd5e13aa26f22a74287abcacbc0.md) | 7,942 |  11,167,961 |  995 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/regex-c7a9ad2ee0e91dd5e13aa26f22a74287abcacbc0.md) | 1,037 |  4,090,656 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/ecrecover-c7a9ad2ee0e91dd5e13aa26f22a74287abcacbc0.md) | 302 |  112,210 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/pairing-c7a9ad2ee0e91dd5e13aa26f22a74287abcacbc0.md) | 449 |  592,827 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2942/kitchen_sink-c7a9ad2ee0e91dd5e13aa26f22a74287abcacbc0.md) | 3,717 |  1,979,971 |  860 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c7a9ad2ee0e91dd5e13aa26f22a74287abcacbc0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28394329266)
