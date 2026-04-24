| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2747/fibonacci-8902898057ae847c87f2f289bc8fc77592b97eea.md) | 3,825 |  12,000,265 |  952 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2747/keccak-8902898057ae847c87f2f289bc8fc77592b97eea.md) | 18,593 |  18,655,329 |  3,311 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2747/sha2_bench-8902898057ae847c87f2f289bc8fc77592b97eea.md) | 9,003 |  14,793,960 |  1,408 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2747/regex-8902898057ae847c87f2f289bc8fc77592b97eea.md) | 1,419 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2747/ecrecover-8902898057ae847c87f2f289bc8fc77592b97eea.md) | 652 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2747/pairing-8902898057ae847c87f2f289bc8fc77592b97eea.md) | 902 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2747/kitchen_sink-8902898057ae847c87f2f289bc8fc77592b97eea.md) | 2,086 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8902898057ae847c87f2f289bc8fc77592b97eea

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24876097498)
