| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2848/fibonacci-b56f72168d758d9b1ff5b2b990cbd3b849680ca2.md) | 3,810 |  12,000,265 |  932 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2848/keccak-b56f72168d758d9b1ff5b2b990cbd3b849680ca2.md) | 18,912 |  18,655,329 |  3,336 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2848/sha2_bench-b56f72168d758d9b1ff5b2b990cbd3b849680ca2.md) | 10,067 |  14,793,960 |  1,443 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2848/regex-b56f72168d758d9b1ff5b2b990cbd3b849680ca2.md) | 1,389 |  4,137,067 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2848/ecrecover-b56f72168d758d9b1ff5b2b990cbd3b849680ca2.md) | 612 |  123,583 |  254 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2848/pairing-b56f72168d758d9b1ff5b2b990cbd3b849680ca2.md) | 886 |  1,745,757 |  263 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2848/kitchen_sink-b56f72168d758d9b1ff5b2b990cbd3b849680ca2.md) | 1,907 |  2,579,903 |  417 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b56f72168d758d9b1ff5b2b990cbd3b849680ca2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27042588540)
