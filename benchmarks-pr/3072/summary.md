| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3072/fibonacci-870df5ade11f0f808ff26411c9acd1b4bd86e2b6.md) | 471 |  4,000,051 |  242 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3072/keccak-870df5ade11f0f808ff26411c9acd1b4bd86e2b6.md) | 7,297 |  14,365,133 |  1,538 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3072/sha2_bench-870df5ade11f0f808ff26411c9acd1b4bd86e2b6.md) | 4,746 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3072/regex-870df5ade11f0f808ff26411c9acd1b4bd86e2b6.md) | 671 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3072/ecrecover-870df5ade11f0f808ff26411c9acd1b4bd86e2b6.md) | 229 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3072/pairing-870df5ade11f0f808ff26411c9acd1b4bd86e2b6.md) | 319 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3072/kitchen_sink-870df5ade11f0f808ff26411c9acd1b4bd86e2b6.md) | 2,679 |  1,979,971 |  471 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/870df5ade11f0f808ff26411c9acd1b4bd86e2b6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30285523705)
