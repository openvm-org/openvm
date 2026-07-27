| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/fibonacci-05d1743309fb283aa09b47e4f8532c288daf2104.md) | 470 |  4,000,051 |  242 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/keccak-05d1743309fb283aa09b47e4f8532c288daf2104.md) | 7,362 |  14,365,133 |  1,536 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/sha2_bench-05d1743309fb283aa09b47e4f8532c288daf2104.md) | 4,765 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/regex-05d1743309fb283aa09b47e4f8532c288daf2104.md) | 668 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/ecrecover-05d1743309fb283aa09b47e4f8532c288daf2104.md) | 222 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/pairing-05d1743309fb283aa09b47e4f8532c288daf2104.md) | 267 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/kitchen_sink-05d1743309fb283aa09b47e4f8532c288daf2104.md) | 2,742 |  1,979,971 |  475 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/05d1743309fb283aa09b47e4f8532c288daf2104

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30289750155)
