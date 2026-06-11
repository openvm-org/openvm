| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2874/fibonacci-ffaeeae9111bf58c619825753fcfe8bdef89f4e7.md) | 4,008 |  12,000,265 |  1,158 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2874/keccak-ffaeeae9111bf58c619825753fcfe8bdef89f4e7.md) | 21,580 |  18,655,329 |  4,557 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2874/sha2_bench-ffaeeae9111bf58c619825753fcfe8bdef89f4e7.md) | 9,539 |  14,793,960 |  1,835 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2874/regex-ffaeeae9111bf58c619825753fcfe8bdef89f4e7.md) | 1,504 |  4,137,067 |  430 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2874/ecrecover-ffaeeae9111bf58c619825753fcfe8bdef89f4e7.md) | 606 |  123,583 |  279 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2874/pairing-ffaeeae9111bf58c619825753fcfe8bdef89f4e7.md) | 931 |  1,745,757 |  295 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2874/kitchen_sink-ffaeeae9111bf58c619825753fcfe8bdef89f4e7.md) | 4,111 |  2,579,903 |  880 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ffaeeae9111bf58c619825753fcfe8bdef89f4e7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27340713930)
