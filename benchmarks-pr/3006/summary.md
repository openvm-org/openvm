| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/fibonacci-c176a23e1cffd8448322b7936a219c5f11454914.md) | 473 |  4,000,051 |  240 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/keccak-c176a23e1cffd8448322b7936a219c5f11454914.md) | 7,252 |  14,365,133 |  1,525 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/sha2_bench-c176a23e1cffd8448322b7936a219c5f11454914.md) | 4,758 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/regex-c176a23e1cffd8448322b7936a219c5f11454914.md) | 662 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/ecrecover-c176a23e1cffd8448322b7936a219c5f11454914.md) | 230 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/pairing-c176a23e1cffd8448322b7936a219c5f11454914.md) | 318 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/kitchen_sink-c176a23e1cffd8448322b7936a219c5f11454914.md) | 2,662 |  1,979,971 |  474 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c176a23e1cffd8448322b7936a219c5f11454914

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29844526183)
