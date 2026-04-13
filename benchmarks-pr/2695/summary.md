| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/fibonacci-2760d333dc9f26ec0889404a50df623153c33676.md) | 3,858 |  12,000,265 |  956 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/keccak-2760d333dc9f26ec0889404a50df623153c33676.md) | 18,606 |  18,655,329 |  3,320 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/sha2_bench-2760d333dc9f26ec0889404a50df623153c33676.md) | 9,027 |  14,793,960 |  1,404 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/regex-2760d333dc9f26ec0889404a50df623153c33676.md) | 1,419 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/ecrecover-2760d333dc9f26ec0889404a50df623153c33676.md) | 644 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/pairing-2760d333dc9f26ec0889404a50df623153c33676.md) | 905 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/kitchen_sink-2760d333dc9f26ec0889404a50df623153c33676.md) | 2,101 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2760d333dc9f26ec0889404a50df623153c33676

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24363466791)
