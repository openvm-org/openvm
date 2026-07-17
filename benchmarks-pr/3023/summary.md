| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/fibonacci-fab72a4f95b35788b2697bd46ead4f80ef0a6bf8.md) | 412 |  4,000,051 |  238 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/keccak-fab72a4f95b35788b2697bd46ead4f80ef0a6bf8.md) | 8,774 |  14,365,133 |  1,546 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/sha2_bench-fab72a4f95b35788b2697bd46ead4f80ef0a6bf8.md) | 4,217 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/regex-fab72a4f95b35788b2697bd46ead4f80ef0a6bf8.md) | 575 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/ecrecover-fab72a4f95b35788b2697bd46ead4f80ef0a6bf8.md) | 234 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/pairing-fab72a4f95b35788b2697bd46ead4f80ef0a6bf8.md) | 300 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/kitchen_sink-fab72a4f95b35788b2697bd46ead4f80ef0a6bf8.md) | 1,926 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fab72a4f95b35788b2697bd46ead4f80ef0a6bf8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29609137447)
