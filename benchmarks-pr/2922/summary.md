| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/fibonacci-d3318bea6681ffd558f507d950fdd3ef2f2d7228.md) | 1,042 |  4,000,051 |  394 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/keccak-d3318bea6681ffd558f507d950fdd3ef2f2d7228.md) | 15,549 |  14,365,133 |  3,039 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/sha2_bench-d3318bea6681ffd558f507d950fdd3ef2f2d7228.md) | 7,772 |  11,167,961 |  999 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/regex-d3318bea6681ffd558f507d950fdd3ef2f2d7228.md) | 1,021 |  4,090,656 |  300 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/ecrecover-d3318bea6681ffd558f507d950fdd3ef2f2d7228.md) | 433 |  112,210 |  285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/pairing-d3318bea6681ffd558f507d950fdd3ef2f2d7228.md) | 543 |  592,827 |  299 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/kitchen_sink-d3318bea6681ffd558f507d950fdd3ef2f2d7228.md) | 3,798 |  1,979,971 |  867 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d3318bea6681ffd558f507d950fdd3ef2f2d7228

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28381561678)
