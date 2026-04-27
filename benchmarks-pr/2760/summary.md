| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2760/fibonacci-6b6779983ea64e0af4494b50d60b8dc41a3546ff.md) | 1,901 |  4,000,051 |  540 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2760/keccak-6b6779983ea64e0af4494b50d60b8dc41a3546ff.md) | 13,479 |  14,365,133 |  2,213 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2760/sha2_bench-6b6779983ea64e0af4494b50d60b8dc41a3546ff.md) | 9,470 |  11,167,961 |  1,272 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2760/regex-6b6779983ea64e0af4494b50d60b8dc41a3546ff.md) | 1,596 |  4,090,656 |  379 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2760/ecrecover-6b6779983ea64e0af4494b50d60b8dc41a3546ff.md) | 647 |  112,210 |  289 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2760/pairing-6b6779983ea64e0af4494b50d60b8dc41a3546ff.md) | 757 |  592,827 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2760/kitchen_sink-6b6779983ea64e0af4494b50d60b8dc41a3546ff.md) | 2,098 |  1,979,971 |  435 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6b6779983ea64e0af4494b50d60b8dc41a3546ff

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25019501288)
