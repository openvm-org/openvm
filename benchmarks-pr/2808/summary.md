| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/fibonacci-a322ef81e1528d4e18030c6d7f7c24fccee735db.md) | 1,584 |  4,000,051 |  437 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/keccak-a322ef81e1528d4e18030c6d7f7c24fccee735db.md) | 13,951 |  14,365,133 |  2,395 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/sha2_bench-a322ef81e1528d4e18030c6d7f7c24fccee735db.md) | 9,371 |  11,167,961 |  1,431 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/regex-a322ef81e1528d4e18030c6d7f7c24fccee735db.md) | 1,464 |  4,090,656 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/ecrecover-a322ef81e1528d4e18030c6d7f7c24fccee735db.md) | 471 |  112,210 |  264 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/pairing-a322ef81e1528d4e18030c6d7f7c24fccee735db.md) | 590 |  592,827 |  252 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/kitchen_sink-a322ef81e1528d4e18030c6d7f7c24fccee735db.md) | 1,823 |  1,979,971 |  408 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a322ef81e1528d4e18030c6d7f7c24fccee735db

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26296393711)
