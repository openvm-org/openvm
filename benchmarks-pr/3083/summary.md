| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/fibonacci-95b5084c7c1fab0c418bf95016c37c912fb9413c.md) | 456 |  4,000,051 |  238 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/keccak-95b5084c7c1fab0c418bf95016c37c912fb9413c.md) | 7,309 |  14,365,133 |  1,532 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/sha2_bench-95b5084c7c1fab0c418bf95016c37c912fb9413c.md) | 4,715 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/regex-95b5084c7c1fab0c418bf95016c37c912fb9413c.md) | 664 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/ecrecover-95b5084c7c1fab0c418bf95016c37c912fb9413c.md) | 230 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/pairing-95b5084c7c1fab0c418bf95016c37c912fb9413c.md) | 300 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/kitchen_sink-95b5084c7c1fab0c418bf95016c37c912fb9413c.md) | 2,643 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/95b5084c7c1fab0c418bf95016c37c912fb9413c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30563346063)
