| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3124/fibonacci-a3646859bdcdd94bef91199806573e31b07eb532.md) | 1,581 |  12,000,265 |  358 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3124/keccak-a3646859bdcdd94bef91199806573e31b07eb532.md) | 9,387 |  18,655,329 |  1,523 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3124/sha2_bench-a3646859bdcdd94bef91199806573e31b07eb532.md) | 4,974 |  14,793,960 |  580 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3124/regex-a3646859bdcdd94bef91199806573e31b07eb532.md) | 668 |  4,137,067 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3124/ecrecover-a3646859bdcdd94bef91199806573e31b07eb532.md) | 426 |  123,583 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3124/pairing-a3646859bdcdd94bef91199806573e31b07eb532.md) | 557 |  1,745,757 |  190 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3124/kitchen_sink-a3646859bdcdd94bef91199806573e31b07eb532.md) | 2,203 |  2,579,903 |  476 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a3646859bdcdd94bef91199806573e31b07eb532

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32393992487)
