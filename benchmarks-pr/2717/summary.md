| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2717/fibonacci-ccdfaba45e096b664bf423c0b821caa6c302c76c.md) | 3,840 |  12,000,265 |  954 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2717/keccak-ccdfaba45e096b664bf423c0b821caa6c302c76c.md) | 18,549 |  18,655,329 |  3,318 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2717/sha2_bench-ccdfaba45e096b664bf423c0b821caa6c302c76c.md) | 8,856 |  14,793,960 |  1,382 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2717/regex-ccdfaba45e096b664bf423c0b821caa6c302c76c.md) | 1,439 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2717/ecrecover-ccdfaba45e096b664bf423c0b821caa6c302c76c.md) | 651 |  123,583 |  277 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2717/pairing-ccdfaba45e096b664bf423c0b821caa6c302c76c.md) | 901 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2717/kitchen_sink-ccdfaba45e096b664bf423c0b821caa6c302c76c.md) | 2,109 |  2,579,903 |  442 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ccdfaba45e096b664bf423c0b821caa6c302c76c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24669249812)
