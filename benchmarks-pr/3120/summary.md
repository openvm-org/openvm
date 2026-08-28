| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/fibonacci-e15273992c6dfbf77e8d288e5e87f744affa3f8b.md) | 487 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/keccak-e15273992c6dfbf77e8d288e5e87f744affa3f8b.md) | 7,661 |  14,365,133 |  1,613 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/sha2_bench-e15273992c6dfbf77e8d288e5e87f744affa3f8b.md) | 4,275 |  11,167,961 |  528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/regex-e15273992c6dfbf77e8d288e5e87f744affa3f8b.md) | 777 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/ecrecover-e15273992c6dfbf77e8d288e5e87f744affa3f8b.md) | 207 |  112,210 |  191 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/pairing-e15273992c6dfbf77e8d288e5e87f744affa3f8b.md) | 255 |  592,827 |  175 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/kitchen_sink-e15273992c6dfbf77e8d288e5e87f744affa3f8b.md) | 2,246 |  1,979,971 |  471 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e15273992c6dfbf77e8d288e5e87f744affa3f8b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33155655332)
