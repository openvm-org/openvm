| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3090/fibonacci-1d75db57bdf49f0a3f14e81832ecb2f9a06d8f51.md) | 481 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3090/keccak-1d75db57bdf49f0a3f14e81832ecb2f9a06d8f51.md) | 7,338 |  14,365,133 |  1,504 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3090/sha2_bench-1d75db57bdf49f0a3f14e81832ecb2f9a06d8f51.md) | 4,136 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3090/regex-1d75db57bdf49f0a3f14e81832ecb2f9a06d8f51.md) | 660 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3090/ecrecover-1d75db57bdf49f0a3f14e81832ecb2f9a06d8f51.md) | 227 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3090/pairing-1d75db57bdf49f0a3f14e81832ecb2f9a06d8f51.md) | 236 |  592,827 |  180 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3090/kitchen_sink-1d75db57bdf49f0a3f14e81832ecb2f9a06d8f51.md) | 2,041 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1d75db57bdf49f0a3f14e81832ecb2f9a06d8f51

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30806729060)
