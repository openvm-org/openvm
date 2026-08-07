| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/fibonacci-330221c06e0dd0590cdbdbad9822fe7f9cfc8d95.md) | 476 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/keccak-330221c06e0dd0590cdbdbad9822fe7f9cfc8d95.md) | 7,327 |  14,365,133 |  1,520 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/sha2_bench-330221c06e0dd0590cdbdbad9822fe7f9cfc8d95.md) | 4,201 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/regex-330221c06e0dd0590cdbdbad9822fe7f9cfc8d95.md) | 664 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/ecrecover-330221c06e0dd0590cdbdbad9822fe7f9cfc8d95.md) | 227 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/pairing-330221c06e0dd0590cdbdbad9822fe7f9cfc8d95.md) | 232 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/kitchen_sink-330221c06e0dd0590cdbdbad9822fe7f9cfc8d95.md) | 2,038 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/330221c06e0dd0590cdbdbad9822fe7f9cfc8d95

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31211760983)
