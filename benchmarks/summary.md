| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-0a19134558ea45ad03fda1d4f5dc4b5e5039052d.md) | 1,709 |  12,000,265 |  374 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-0a19134558ea45ad03fda1d4f5dc4b5e5039052d.md) | 9,579 |  18,655,329 |  1,548 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-0a19134558ea45ad03fda1d4f5dc4b5e5039052d.md) | 5,247 |  14,793,960 |  588 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-0a19134558ea45ad03fda1d4f5dc4b5e5039052d.md) | 705 |  4,137,067 |  223 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-0a19134558ea45ad03fda1d4f5dc4b5e5039052d.md) | 432 |  123,583 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-0a19134558ea45ad03fda1d4f5dc4b5e5039052d.md) | 611 |  1,745,757 |  193 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-0a19134558ea45ad03fda1d4f5dc4b5e5039052d.md) | 2,306 |  2,579,903 |  493 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0a19134558ea45ad03fda1d4f5dc4b5e5039052d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/38796970061)
