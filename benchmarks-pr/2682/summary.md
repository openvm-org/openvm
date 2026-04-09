| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/fibonacci-7f4d7b7bd1b1de59c971a50c890705ed05f1f7e5.md) | 3,783 |  12,000,265 |  937 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/keccak-7f4d7b7bd1b1de59c971a50c890705ed05f1f7e5.md) | 18,761 |  18,655,329 |  3,352 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/regex-7f4d7b7bd1b1de59c971a50c890705ed05f1f7e5.md) | 1,421 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/ecrecover-7f4d7b7bd1b1de59c971a50c890705ed05f1f7e5.md) | 646 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/pairing-7f4d7b7bd1b1de59c971a50c890705ed05f1f7e5.md) | 909 |  1,745,757 |  287 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/kitchen_sink-7f4d7b7bd1b1de59c971a50c890705ed05f1f7e5.md) | 2,147 |  2,579,903 |  431 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7f4d7b7bd1b1de59c971a50c890705ed05f1f7e5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24199189210)
