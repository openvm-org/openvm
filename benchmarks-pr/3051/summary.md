| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/fibonacci-2017743cc06138565eeeb2b29323b5dc6d37a31a.md) | 463 |  4,000,051 |  242 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/keccak-2017743cc06138565eeeb2b29323b5dc6d37a31a.md) | 7,321 |  14,365,133 |  1,533 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/sha2_bench-2017743cc06138565eeeb2b29323b5dc6d37a31a.md) | 4,753 |  11,167,961 |  532 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/regex-2017743cc06138565eeeb2b29323b5dc6d37a31a.md) | 679 |  4,090,656 |  220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/ecrecover-2017743cc06138565eeeb2b29323b5dc6d37a31a.md) | 230 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/pairing-2017743cc06138565eeeb2b29323b5dc6d37a31a.md) | 312 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/kitchen_sink-2017743cc06138565eeeb2b29323b5dc6d37a31a.md) | 2,654 |  1,979,971 |  466 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2017743cc06138565eeeb2b29323b5dc6d37a31a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30105534990)
