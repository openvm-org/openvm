| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2894/fibonacci-fad04b4ea10925f7c55253c6208d5679ab6aba06.md) | 3,130 |  12,000,265 |  687 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2894/keccak-fad04b4ea10925f7c55253c6208d5679ab6aba06.md) | 16,473 |  18,655,329 |  3,056 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2894/sha2_bench-fad04b4ea10925f7c55253c6208d5679ab6aba06.md) | 9,285 |  14,793,960 |  1,138 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2894/regex-fad04b4ea10925f7c55253c6208d5679ab6aba06.md) | 1,173 |  4,137,067 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2894/ecrecover-fad04b4ea10925f7c55253c6208d5679ab6aba06.md) | 596 |  123,583 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2894/pairing-fad04b4ea10925f7c55253c6208d5679ab6aba06.md) | 947 |  1,745,757 |  295 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2894/kitchen_sink-fad04b4ea10925f7c55253c6208d5679ab6aba06.md) | 4,117 |  2,579,903 |  885 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fad04b4ea10925f7c55253c6208d5679ab6aba06

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27618236129)
