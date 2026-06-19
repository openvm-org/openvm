| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2907/fibonacci-fb301faa3012dfcd4fe98f7238a916c2f3bfd504.md) | 3,057 |  12,000,265 |  675 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2907/keccak-fb301faa3012dfcd4fe98f7238a916c2f3bfd504.md) | 16,407 |  18,655,329 |  3,036 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2907/sha2_bench-fb301faa3012dfcd4fe98f7238a916c2f3bfd504.md) | 9,198 |  14,793,960 |  1,124 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2907/regex-fb301faa3012dfcd4fe98f7238a916c2f3bfd504.md) | 1,181 |  4,137,067 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2907/ecrecover-fb301faa3012dfcd4fe98f7238a916c2f3bfd504.md) | 608 |  123,583 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2907/pairing-fb301faa3012dfcd4fe98f7238a916c2f3bfd504.md) | 935 |  1,745,757 |  307 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2907/kitchen_sink-fb301faa3012dfcd4fe98f7238a916c2f3bfd504.md) | 4,088 |  2,579,903 |  871 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fb301faa3012dfcd4fe98f7238a916c2f3bfd504

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27819653846)
