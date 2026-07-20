| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/fibonacci-c692c2d8817574c24b999de9c8a3443eba570e1a.md) | 411 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/keccak-c692c2d8817574c24b999de9c8a3443eba570e1a.md) | 8,670 |  14,365,133 |  1,532 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/sha2_bench-c692c2d8817574c24b999de9c8a3443eba570e1a.md) | 4,200 |  11,167,961 |  518 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/regex-c692c2d8817574c24b999de9c8a3443eba570e1a.md) | 579 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/ecrecover-c692c2d8817574c24b999de9c8a3443eba570e1a.md) | 218 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/pairing-c692c2d8817574c24b999de9c8a3443eba570e1a.md) | 286 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/kitchen_sink-c692c2d8817574c24b999de9c8a3443eba570e1a.md) | 1,926 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c692c2d8817574c24b999de9c8a3443eba570e1a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29761017704)
