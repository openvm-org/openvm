| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/fibonacci-f4d7c74cec44c657746e9d629b81991173e0b333.md) | 481 |  4,000,051 |  238 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/keccak-f4d7c74cec44c657746e9d629b81991173e0b333.md) | 7,636 |  14,365,133 |  1,611 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/sha2_bench-f4d7c74cec44c657746e9d629b81991173e0b333.md) | 4,347 |  11,167,961 |  531 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/regex-f4d7c74cec44c657746e9d629b81991173e0b333.md) | 763 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/ecrecover-f4d7c74cec44c657746e9d629b81991173e0b333.md) | 210 |  112,210 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/pairing-f4d7c74cec44c657746e9d629b81991173e0b333.md) | 247 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/kitchen_sink-f4d7c74cec44c657746e9d629b81991173e0b333.md) | 2,257 |  1,979,971 |  477 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f4d7c74cec44c657746e9d629b81991173e0b333

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/38980355179)
