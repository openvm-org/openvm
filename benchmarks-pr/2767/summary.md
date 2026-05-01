| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/fibonacci-249a438513b7ba75f0d5bce0f75d5a82162f196b.md) | 3,880 |  12,000,265 |  968 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/keccak-249a438513b7ba75f0d5bce0f75d5a82162f196b.md) | 18,348 |  18,655,329 |  3,283 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/sha2_bench-249a438513b7ba75f0d5bce0f75d5a82162f196b.md) | 8,955 |  14,793,960 |  1,396 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/regex-249a438513b7ba75f0d5bce0f75d5a82162f196b.md) | 1,423 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/ecrecover-249a438513b7ba75f0d5bce0f75d5a82162f196b.md) | 645 |  123,583 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/pairing-249a438513b7ba75f0d5bce0f75d5a82162f196b.md) | 896 |  1,745,757 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2767/kitchen_sink-249a438513b7ba75f0d5bce0f75d5a82162f196b.md) | 2,074 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/249a438513b7ba75f0d5bce0f75d5a82162f196b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25206549016)
