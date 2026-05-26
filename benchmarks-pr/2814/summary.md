| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2814/fibonacci-a1e2fb55df7364f4186b458ccbd6424d4981b439.md) | 3,711 |  12,000,265 |  906 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2814/keccak-a1e2fb55df7364f4186b458ccbd6424d4981b439.md) | 18,594 |  18,655,329 |  3,273 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2814/sha2_bench-a1e2fb55df7364f4186b458ccbd6424d4981b439.md) | 9,990 |  14,793,960 |  1,435 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2814/regex-a1e2fb55df7364f4186b458ccbd6424d4981b439.md) | 1,410 |  4,137,067 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2814/ecrecover-a1e2fb55df7364f4186b458ccbd6424d4981b439.md) | 603 |  123,583 |  251 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2814/pairing-a1e2fb55df7364f4186b458ccbd6424d4981b439.md) | 889 |  1,745,757 |  263 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2814/kitchen_sink-a1e2fb55df7364f4186b458ccbd6424d4981b439.md) | 1,900 |  2,579,903 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a1e2fb55df7364f4186b458ccbd6424d4981b439

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26465985034)
