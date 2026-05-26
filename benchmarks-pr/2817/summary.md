| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/fibonacci-c56a3a5ea3cbf3103f3941a8ae4035f581cccb9c.md) | 3,777 |  12,000,265 |  921 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/keccak-c56a3a5ea3cbf3103f3941a8ae4035f581cccb9c.md) | 18,475 |  18,655,329 |  3,260 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/sha2_bench-c56a3a5ea3cbf3103f3941a8ae4035f581cccb9c.md) | 10,179 |  14,793,960 |  1,466 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/regex-c56a3a5ea3cbf3103f3941a8ae4035f581cccb9c.md) | 1,400 |  4,137,067 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/ecrecover-c56a3a5ea3cbf3103f3941a8ae4035f581cccb9c.md) | 596 |  123,583 |  246 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/pairing-c56a3a5ea3cbf3103f3941a8ae4035f581cccb9c.md) | 890 |  1,745,757 |  263 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/kitchen_sink-c56a3a5ea3cbf3103f3941a8ae4035f581cccb9c.md) | 1,908 |  2,579,903 |  417 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c56a3a5ea3cbf3103f3941a8ae4035f581cccb9c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26471448967)
