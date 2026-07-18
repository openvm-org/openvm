| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/fibonacci-02fc053c84b9af081165987dd5c5cec002232177.md) | 418 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/keccak-02fc053c84b9af081165987dd5c5cec002232177.md) | 8,590 |  14,365,133 |  1,516 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/sha2_bench-02fc053c84b9af081165987dd5c5cec002232177.md) | 4,224 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/regex-02fc053c84b9af081165987dd5c5cec002232177.md) | 575 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/ecrecover-02fc053c84b9af081165987dd5c5cec002232177.md) | 217 |  112,210 |  179 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/pairing-02fc053c84b9af081165987dd5c5cec002232177.md) | 282 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/kitchen_sink-02fc053c84b9af081165987dd5c5cec002232177.md) | 1,917 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/02fc053c84b9af081165987dd5c5cec002232177

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29652769446)
