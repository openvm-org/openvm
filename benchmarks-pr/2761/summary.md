| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2761/fibonacci-b0e04226e15174d496b64d23d809bde1fac1a38b.md) | 1,886 |  4,000,051 |  536 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2761/keccak-b0e04226e15174d496b64d23d809bde1fac1a38b.md) | 13,599 |  14,365,133 |  2,239 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2761/sha2_bench-b0e04226e15174d496b64d23d809bde1fac1a38b.md) | 9,661 |  11,167,961 |  1,294 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2761/regex-b0e04226e15174d496b64d23d809bde1fac1a38b.md) | 1,609 |  4,090,656 |  383 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2761/ecrecover-b0e04226e15174d496b64d23d809bde1fac1a38b.md) | 650 |  112,210 |  292 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2761/pairing-b0e04226e15174d496b64d23d809bde1fac1a38b.md) | 772 |  592,827 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2761/kitchen_sink-b0e04226e15174d496b64d23d809bde1fac1a38b.md) | 2,064 |  1,979,971 |  429 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b0e04226e15174d496b64d23d809bde1fac1a38b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25059053191)
