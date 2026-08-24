| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/fibonacci-1d733a9369a0fd0cf8a465d2aadd3cf7f7986244.md) | 459 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/keccak-1d733a9369a0fd0cf8a465d2aadd3cf7f7986244.md) | 7,186 |  14,365,133 |  1,572 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/sha2_bench-1d733a9369a0fd0cf8a465d2aadd3cf7f7986244.md) | 4,067 |  11,167,961 |  518 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/regex-1d733a9369a0fd0cf8a465d2aadd3cf7f7986244.md) | 732 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/ecrecover-1d733a9369a0fd0cf8a465d2aadd3cf7f7986244.md) | 207 |  112,210 |  189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/pairing-1d733a9369a0fd0cf8a465d2aadd3cf7f7986244.md) | 240 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/kitchen_sink-1d733a9369a0fd0cf8a465d2aadd3cf7f7986244.md) | 2,135 |  1,979,971 |  457 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1d733a9369a0fd0cf8a465d2aadd3cf7f7986244

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32724614750)
