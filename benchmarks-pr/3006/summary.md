| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/fibonacci-d445fc654d17446a299ed5b23b34aafe39ca93f6.md) | 471 |  4,000,051 |  239 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/keccak-d445fc654d17446a299ed5b23b34aafe39ca93f6.md) | 7,262 |  14,365,133 |  1,533 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/sha2_bench-d445fc654d17446a299ed5b23b34aafe39ca93f6.md) | 4,663 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/regex-d445fc654d17446a299ed5b23b34aafe39ca93f6.md) | 659 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/ecrecover-d445fc654d17446a299ed5b23b34aafe39ca93f6.md) | 229 |  112,210 |  189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/pairing-d445fc654d17446a299ed5b23b34aafe39ca93f6.md) | 313 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/kitchen_sink-d445fc654d17446a299ed5b23b34aafe39ca93f6.md) | 2,669 |  1,979,971 |  475 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d445fc654d17446a299ed5b23b34aafe39ca93f6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29831784582)
