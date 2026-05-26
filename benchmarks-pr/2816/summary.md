| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/fibonacci-ad2a8647503020f5ecb0888622ff303c34e3fe6c.md) | 1,890 |  4,000,051 |  514 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/keccak-ad2a8647503020f5ecb0888622ff303c34e3fe6c.md) | 13,571 |  14,365,133 |  2,217 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/sha2_bench-ad2a8647503020f5ecb0888622ff303c34e3fe6c.md) | 9,472 |  11,167,961 |  1,408 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/regex-ad2a8647503020f5ecb0888622ff303c34e3fe6c.md) | 1,570 |  4,090,656 |  366 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/ecrecover-ad2a8647503020f5ecb0888622ff303c34e3fe6c.md) | 607 |  112,210 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/pairing-ad2a8647503020f5ecb0888622ff303c34e3fe6c.md) | 751 |  592,827 |  265 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2816/kitchen_sink-ad2a8647503020f5ecb0888622ff303c34e3fe6c.md) | 1,871 |  1,979,971 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ad2a8647503020f5ecb0888622ff303c34e3fe6c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26475467628)
