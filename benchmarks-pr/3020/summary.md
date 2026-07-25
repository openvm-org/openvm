| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-011b04d9c20ea59839f6b06ebdc7c579dd3e797b.md) | 481 |  4,000,051 |  239 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-011b04d9c20ea59839f6b06ebdc7c579dd3e797b.md) | 10,369 |  14,365,133 |  1,549 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-011b04d9c20ea59839f6b06ebdc7c579dd3e797b.md) | 4,649 |  11,167,961 |  530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-011b04d9c20ea59839f6b06ebdc7c579dd3e797b.md) | 686 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-011b04d9c20ea59839f6b06ebdc7c579dd3e797b.md) | 231 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-011b04d9c20ea59839f6b06ebdc7c579dd3e797b.md) | 275 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-011b04d9c20ea59839f6b06ebdc7c579dd3e797b.md) | 2,368 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/011b04d9c20ea59839f6b06ebdc7c579dd3e797b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30138311459)
