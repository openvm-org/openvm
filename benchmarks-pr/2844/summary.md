| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2844/fibonacci-6be74102e7b9caab06777c321b7d9d9b0a8d9b78.md) | 3,792 |  12,000,265 |  929 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2844/keccak-6be74102e7b9caab06777c321b7d9d9b0a8d9b78.md) | 18,234 |  18,655,329 |  3,312 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2844/sha2_bench-6be74102e7b9caab06777c321b7d9d9b0a8d9b78.md) | 9,908 |  14,793,960 |  1,455 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2844/regex-6be74102e7b9caab06777c321b7d9d9b0a8d9b78.md) | 1,393 |  4,137,067 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2844/ecrecover-6be74102e7b9caab06777c321b7d9d9b0a8d9b78.md) | 600 |  123,583 |  253 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2844/pairing-6be74102e7b9caab06777c321b7d9d9b0a8d9b78.md) | 882 |  1,745,757 |  264 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2844/kitchen_sink-6be74102e7b9caab06777c321b7d9d9b0a8d9b78.md) | 3,850 |  2,579,903 |  946 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6be74102e7b9caab06777c321b7d9d9b0a8d9b78

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27019738658)
