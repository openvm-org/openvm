| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-07790a83a18763506c373a0140ec66f76cd89bb6.md) | 1,660 |  12,000,265 |  366 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-07790a83a18763506c373a0140ec66f76cd89bb6.md) | 9,544 |  18,655,329 |  1,523 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-07790a83a18763506c373a0140ec66f76cd89bb6.md) | 5,349 |  14,793,960 |  593 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-07790a83a18763506c373a0140ec66f76cd89bb6.md) | 701 |  4,137,067 |  220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-07790a83a18763506c373a0140ec66f76cd89bb6.md) | 441 |  123,583 |  190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-07790a83a18763506c373a0140ec66f76cd89bb6.md) | 630 |  1,745,757 |  199 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-07790a83a18763506c373a0140ec66f76cd89bb6.md) | 2,290 |  2,579,903 |  491 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/07790a83a18763506c373a0140ec66f76cd89bb6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/40365161102)
