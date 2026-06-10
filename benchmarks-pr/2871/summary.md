| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/fibonacci-6ac55b49cce58586ba8f54bf96417f8ead2a9b1a.md) | 4,017 |  12,000,265 |  1,155 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/keccak-6ac55b49cce58586ba8f54bf96417f8ead2a9b1a.md) | 21,555 |  18,655,329 |  4,572 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/sha2_bench-6ac55b49cce58586ba8f54bf96417f8ead2a9b1a.md) | 9,611 |  14,793,960 |  1,839 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/regex-6ac55b49cce58586ba8f54bf96417f8ead2a9b1a.md) | 1,505 |  4,137,067 |  429 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/ecrecover-6ac55b49cce58586ba8f54bf96417f8ead2a9b1a.md) | 616 |  123,583 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/pairing-6ac55b49cce58586ba8f54bf96417f8ead2a9b1a.md) | 954 |  1,745,757 |  309 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2871/kitchen_sink-6ac55b49cce58586ba8f54bf96417f8ead2a9b1a.md) | 4,148 |  2,579,903 |  889 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6ac55b49cce58586ba8f54bf96417f8ead2a9b1a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27313080566)
