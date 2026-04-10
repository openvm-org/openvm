| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/fibonacci-ab47c058aa01a21031247ed7d25bf46df98d8db5.md) | 3,923 |  12,000,265 |  975 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/keccak-ab47c058aa01a21031247ed7d25bf46df98d8db5.md) | 18,439 |  18,655,329 |  3,304 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/sha2_bench-ab47c058aa01a21031247ed7d25bf46df98d8db5.md) | 9,852 |  14,793,960 |  1,391 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/regex-ab47c058aa01a21031247ed7d25bf46df98d8db5.md) | 1,413 |  4,137,067 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/ecrecover-ab47c058aa01a21031247ed7d25bf46df98d8db5.md) | 651 |  123,583 |  278 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/pairing-ab47c058aa01a21031247ed7d25bf46df98d8db5.md) | 913 |  1,745,757 |  287 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/kitchen_sink-ab47c058aa01a21031247ed7d25bf46df98d8db5.md) | 2,174 |  2,579,903 |  439 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ab47c058aa01a21031247ed7d25bf46df98d8db5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24265897298)
