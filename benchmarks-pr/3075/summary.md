| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3075/fibonacci-dac796e3b0179c2dd869c82c4c5b908bfded5606.md) | 472 |  4,000,051 |  238 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3075/keccak-dac796e3b0179c2dd869c82c4c5b908bfded5606.md) | 7,335 |  14,365,133 |  1,533 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3075/sha2_bench-dac796e3b0179c2dd869c82c4c5b908bfded5606.md) | 4,764 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3075/regex-dac796e3b0179c2dd869c82c4c5b908bfded5606.md) | 668 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3075/ecrecover-dac796e3b0179c2dd869c82c4c5b908bfded5606.md) | 232 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3075/pairing-dac796e3b0179c2dd869c82c4c5b908bfded5606.md) | 325 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3075/kitchen_sink-dac796e3b0179c2dd869c82c4c5b908bfded5606.md) | 2,660 |  1,979,971 |  471 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/dac796e3b0179c2dd869c82c4c5b908bfded5606

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30304176854)
