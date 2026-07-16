| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3028/fibonacci-8298dfafaea415e0897fef6270d41d9fb09956be.md) | 415 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3028/keccak-8298dfafaea415e0897fef6270d41d9fb09956be.md) | 8,515 |  14,365,133 |  1,545 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3028/sha2_bench-8298dfafaea415e0897fef6270d41d9fb09956be.md) | 3,938 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3028/regex-8298dfafaea415e0897fef6270d41d9fb09956be.md) | 574 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3028/ecrecover-8298dfafaea415e0897fef6270d41d9fb09956be.md) | 216 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3028/pairing-8298dfafaea415e0897fef6270d41d9fb09956be.md) | 276 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3028/kitchen_sink-8298dfafaea415e0897fef6270d41d9fb09956be.md) | 1,893 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8298dfafaea415e0897fef6270d41d9fb09956be

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29498519277)
