| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/fibonacci-770c77f78e96b650ba9f761a96ce34c23634895c.md) | 475 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/keccak-770c77f78e96b650ba9f761a96ce34c23634895c.md) | 7,398 |  14,365,133 |  1,540 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/sha2_bench-770c77f78e96b650ba9f761a96ce34c23634895c.md) | 4,110 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/regex-770c77f78e96b650ba9f761a96ce34c23634895c.md) | 655 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/ecrecover-770c77f78e96b650ba9f761a96ce34c23634895c.md) | 230 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/pairing-770c77f78e96b650ba9f761a96ce34c23634895c.md) | 230 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/kitchen_sink-770c77f78e96b650ba9f761a96ce34c23634895c.md) | 2,048 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/770c77f78e96b650ba9f761a96ce34c23634895c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31036716258)
