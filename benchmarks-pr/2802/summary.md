| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/fibonacci-e974555e6493d4e9b66f697ffecc4ea78bd82f16.md) | 1,576 |  4,000,051 |  429 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/keccak-e974555e6493d4e9b66f697ffecc4ea78bd82f16.md) | 14,091 |  14,365,133 |  2,373 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/sha2_bench-e974555e6493d4e9b66f697ffecc4ea78bd82f16.md) | 9,141 |  11,167,961 |  1,395 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/regex-e974555e6493d4e9b66f697ffecc4ea78bd82f16.md) | 1,598 |  4,090,656 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/ecrecover-e974555e6493d4e9b66f697ffecc4ea78bd82f16.md) | 487 |  112,210 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/pairing-e974555e6493d4e9b66f697ffecc4ea78bd82f16.md) | 601 |  592,827 |  253 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/kitchen_sink-e974555e6493d4e9b66f697ffecc4ea78bd82f16.md) | 2,171 |  1,979,971 |  411 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e974555e6493d4e9b66f697ffecc4ea78bd82f16

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26878043810)
