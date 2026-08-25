| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-995b2dc0002d514abcba0f0b2fd4a0a8c68f031f.md) | 1,700 |  12,000,265 |  372 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-995b2dc0002d514abcba0f0b2fd4a0a8c68f031f.md) | 9,722 |  18,655,329 |  1,540 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-995b2dc0002d514abcba0f0b2fd4a0a8c68f031f.md) | 5,337 |  14,793,960 |  594 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-995b2dc0002d514abcba0f0b2fd4a0a8c68f031f.md) | 693 |  4,137,067 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-995b2dc0002d514abcba0f0b2fd4a0a8c68f031f.md) | 428 |  123,583 |  189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-995b2dc0002d514abcba0f0b2fd4a0a8c68f031f.md) | 579 |  1,745,757 |  196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-995b2dc0002d514abcba0f0b2fd4a0a8c68f031f.md) | 2,283 |  2,579,903 |  489 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/995b2dc0002d514abcba0f0b2fd4a0a8c68f031f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32866894686)
