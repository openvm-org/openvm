| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-d42ca4de3ec319324b4d8b1977757d6b585d4b4b.md) | 1,684 |  12,000,265 |  370 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-d42ca4de3ec319324b4d8b1977757d6b585d4b4b.md) | 9,518 |  18,655,329 |  1,564 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-d42ca4de3ec319324b4d8b1977757d6b585d4b4b.md) | 5,357 |  14,793,960 |  586 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-d42ca4de3ec319324b4d8b1977757d6b585d4b4b.md) | 697 |  4,137,067 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-d42ca4de3ec319324b4d8b1977757d6b585d4b4b.md) | 430 |  123,583 |  189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-d42ca4de3ec319324b4d8b1977757d6b585d4b4b.md) | 590 |  1,745,757 |  193 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-d42ca4de3ec319324b4d8b1977757d6b585d4b4b.md) | 2,300 |  2,579,903 |  492 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d42ca4de3ec319324b4d8b1977757d6b585d4b4b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32982029957)
