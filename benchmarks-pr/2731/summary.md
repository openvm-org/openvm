| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2731/fibonacci-92d6d7ea944cf7f77f281e42caf7553d24a98378.md) | 3,764 |  12,000,265 |  937 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2731/keccak-92d6d7ea944cf7f77f281e42caf7553d24a98378.md) | 18,425 |  18,655,329 |  3,293 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2731/sha2_bench-92d6d7ea944cf7f77f281e42caf7553d24a98378.md) | 8,941 |  14,793,960 |  1,395 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2731/regex-92d6d7ea944cf7f77f281e42caf7553d24a98378.md) | 1,447 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2731/ecrecover-92d6d7ea944cf7f77f281e42caf7553d24a98378.md) | 642 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2731/pairing-92d6d7ea944cf7f77f281e42caf7553d24a98378.md) | 905 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2731/kitchen_sink-92d6d7ea944cf7f77f281e42caf7553d24a98378.md) | 2,105 |  2,579,903 |  439 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/92d6d7ea944cf7f77f281e42caf7553d24a98378

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24738078647)
