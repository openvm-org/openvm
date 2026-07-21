| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/fibonacci-f02a88c44be9e646ffad3a8119d3b4d5e34210a5.md) | 472 |  4,000,051 |  245 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/keccak-f02a88c44be9e646ffad3a8119d3b4d5e34210a5.md) | 7,336 |  14,365,133 |  1,557 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/sha2_bench-f02a88c44be9e646ffad3a8119d3b4d5e34210a5.md) | 4,760 |  11,167,961 |  533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/regex-f02a88c44be9e646ffad3a8119d3b4d5e34210a5.md) | 677 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/ecrecover-f02a88c44be9e646ffad3a8119d3b4d5e34210a5.md) | 227 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/pairing-f02a88c44be9e646ffad3a8119d3b4d5e34210a5.md) | 318 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/kitchen_sink-f02a88c44be9e646ffad3a8119d3b4d5e34210a5.md) | 2,604 |  1,979,971 |  471 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f02a88c44be9e646ffad3a8119d3b4d5e34210a5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29790439499)
