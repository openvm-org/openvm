| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/fibonacci-e443c6277898b7c8a859d06bc52f42e822aa6012.md) | 3,798 |  12,000,265 |  926 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/keccak-e443c6277898b7c8a859d06bc52f42e822aa6012.md) | 18,147 |  18,655,329 |  3,295 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/sha2_bench-e443c6277898b7c8a859d06bc52f42e822aa6012.md) | 9,815 |  14,793,960 |  1,433 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/regex-e443c6277898b7c8a859d06bc52f42e822aa6012.md) | 1,394 |  4,137,067 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/ecrecover-e443c6277898b7c8a859d06bc52f42e822aa6012.md) | 614 |  123,583 |  250 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/pairing-e443c6277898b7c8a859d06bc52f42e822aa6012.md) | 885 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/kitchen_sink-e443c6277898b7c8a859d06bc52f42e822aa6012.md) | 1,883 |  2,579,903 |  416 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e443c6277898b7c8a859d06bc52f42e822aa6012

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27556417706)
