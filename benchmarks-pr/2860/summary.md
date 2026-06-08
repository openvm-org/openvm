| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/fibonacci-f0801fc819ed0ea388234dccaa2757099b25ef2f.md) | 3,688 |  12,000,265 |  904 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/keccak-f0801fc819ed0ea388234dccaa2757099b25ef2f.md) | 18,016 |  18,655,329 |  3,275 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/sha2_bench-f0801fc819ed0ea388234dccaa2757099b25ef2f.md) | 9,966 |  14,793,960 |  1,449 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/regex-f0801fc819ed0ea388234dccaa2757099b25ef2f.md) | 1,395 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/ecrecover-f0801fc819ed0ea388234dccaa2757099b25ef2f.md) | 596 |  123,583 |  251 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/pairing-f0801fc819ed0ea388234dccaa2757099b25ef2f.md) | 885 |  1,745,757 |  265 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2860/kitchen_sink-f0801fc819ed0ea388234dccaa2757099b25ef2f.md) | 3,928 |  2,579,903 |  975 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f0801fc819ed0ea388234dccaa2757099b25ef2f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27166072254)
