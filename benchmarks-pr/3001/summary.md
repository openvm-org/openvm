| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3001/fibonacci-80ba513e14c1345c7ce0cc097da476a9771b29df.md) | 3,059 |  12,000,265 |  685 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3001/keccak-80ba513e14c1345c7ce0cc097da476a9771b29df.md) | 16,384 |  18,655,329 |  3,023 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3001/sha2_bench-80ba513e14c1345c7ce0cc097da476a9771b29df.md) | 9,515 |  14,793,960 |  1,133 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3001/regex-80ba513e14c1345c7ce0cc097da476a9771b29df.md) | 1,209 |  4,137,067 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3001/ecrecover-80ba513e14c1345c7ce0cc097da476a9771b29df.md) | 513 |  123,583 |  286 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3001/pairing-80ba513e14c1345c7ce0cc097da476a9771b29df.md) | 845 |  1,745,757 |  312 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3001/kitchen_sink-80ba513e14c1345c7ce0cc097da476a9771b29df.md) | 4,531 |  2,579,903 |  883 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/80ba513e14c1345c7ce0cc097da476a9771b29df

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29074823239)
