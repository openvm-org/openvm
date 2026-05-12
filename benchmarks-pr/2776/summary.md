| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/fibonacci-2eea8924ab85e9c66a9e4173e49ea561bc7ca04b.md) | 3,863 |  12,000,265 |  961 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/keccak-2eea8924ab85e9c66a9e4173e49ea561bc7ca04b.md) | 18,656 |  18,655,329 |  3,328 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/sha2_bench-2eea8924ab85e9c66a9e4173e49ea561bc7ca04b.md) | 8,962 |  14,793,960 |  1,391 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/regex-2eea8924ab85e9c66a9e4173e49ea561bc7ca04b.md) | 1,403 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/ecrecover-2eea8924ab85e9c66a9e4173e49ea561bc7ca04b.md) | 631 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/pairing-2eea8924ab85e9c66a9e4173e49ea561bc7ca04b.md) | 891 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2776/kitchen_sink-2eea8924ab85e9c66a9e4173e49ea561bc7ca04b.md) | 2,100 |  2,579,903 |  438 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2eea8924ab85e9c66a9e4173e49ea561bc7ca04b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25742910196)
