| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/fibonacci-fdf3f2cfd65e7096c0496a08a473c24d7a285c01.md) | 465 |  4,000,051 |  240 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/keccak-fdf3f2cfd65e7096c0496a08a473c24d7a285c01.md) | 7,276 |  14,365,133 |  1,543 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/sha2_bench-fdf3f2cfd65e7096c0496a08a473c24d7a285c01.md) | 4,739 |  11,167,961 |  531 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/regex-fdf3f2cfd65e7096c0496a08a473c24d7a285c01.md) | 654 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/ecrecover-fdf3f2cfd65e7096c0496a08a473c24d7a285c01.md) | 226 |  112,210 |  191 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/pairing-fdf3f2cfd65e7096c0496a08a473c24d7a285c01.md) | 297 |  592,827 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/kitchen_sink-fdf3f2cfd65e7096c0496a08a473c24d7a285c01.md) | 2,655 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fdf3f2cfd65e7096c0496a08a473c24d7a285c01

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30483688438)
