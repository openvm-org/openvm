| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-d12645da582164aed80d0eec38e7af2053d64aea.md) | 457 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-d12645da582164aed80d0eec38e7af2053d64aea.md) | 7,458 |  14,365,133 |  1,535 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-d12645da582164aed80d0eec38e7af2053d64aea.md) | 4,165 |  11,167,961 |  518 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-d12645da582164aed80d0eec38e7af2053d64aea.md) | 679 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-d12645da582164aed80d0eec38e7af2053d64aea.md) | 195 |  112,210 |  203 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-d12645da582164aed80d0eec38e7af2053d64aea.md) | 235 |  592,827 |  204 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-d12645da582164aed80d0eec38e7af2053d64aea.md) | 2,015 |  1,979,971 |  523 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d12645da582164aed80d0eec38e7af2053d64aea

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31694081833)
