| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2997/fibonacci-54802a26beecc6f90f7b26a178b88f25180107a5.md) | 3,022 |  12,000,265 |  679 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2997/keccak-54802a26beecc6f90f7b26a178b88f25180107a5.md) | 16,504 |  18,655,329 |  3,046 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2997/sha2_bench-54802a26beecc6f90f7b26a178b88f25180107a5.md) | 9,493 |  14,793,960 |  1,130 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2997/regex-54802a26beecc6f90f7b26a178b88f25180107a5.md) | 1,211 |  4,137,067 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2997/ecrecover-54802a26beecc6f90f7b26a178b88f25180107a5.md) | 512 |  123,583 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2997/pairing-54802a26beecc6f90f7b26a178b88f25180107a5.md) | 852 |  1,745,757 |  317 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2997/kitchen_sink-54802a26beecc6f90f7b26a178b88f25180107a5.md) | 4,500 |  2,579,903 |  877 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/54802a26beecc6f90f7b26a178b88f25180107a5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29064848381)
