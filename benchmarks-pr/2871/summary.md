| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/fibonacci-ab396ffa148186d3a58a67891c3622f60058df65.md) | 3,936 |  12,000,265 |  1,136 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/keccak-ab396ffa148186d3a58a67891c3622f60058df65.md) | 22,180 |  18,655,329 |  4,700 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/sha2_bench-ab396ffa148186d3a58a67891c3622f60058df65.md) | 9,646 |  14,793,960 |  1,842 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/regex-ab396ffa148186d3a58a67891c3622f60058df65.md) | 1,514 |  4,137,067 |  428 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/ecrecover-ab396ffa148186d3a58a67891c3622f60058df65.md) | 605 |  123,583 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/pairing-ab396ffa148186d3a58a67891c3622f60058df65.md) | 932 |  1,745,757 |  303 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/kitchen_sink-ab396ffa148186d3a58a67891c3622f60058df65.md) | 4,136 |  2,579,903 |  879 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ab396ffa148186d3a58a67891c3622f60058df65

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27314626242)
