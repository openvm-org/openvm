| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3036/fibonacci-965b23403f5f9fbd4d52b6bb4427638fee98b56c.md) | 408 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3036/keccak-965b23403f5f9fbd4d52b6bb4427638fee98b56c.md) | 8,707 |  14,365,133 |  1,536 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3036/sha2_bench-965b23403f5f9fbd4d52b6bb4427638fee98b56c.md) | 4,245 |  11,167,961 |  527 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3036/regex-965b23403f5f9fbd4d52b6bb4427638fee98b56c.md) | 573 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3036/ecrecover-965b23403f5f9fbd4d52b6bb4427638fee98b56c.md) | 216 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3036/pairing-965b23403f5f9fbd4d52b6bb4427638fee98b56c.md) | 292 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3036/kitchen_sink-965b23403f5f9fbd4d52b6bb4427638fee98b56c.md) | 1,926 |  1,979,971 |  466 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3036/fibonacci_e2e-965b23403f5f9fbd4d52b6bb4427638fee98b56c.md) | 426 |  4,000,051 |  223 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3036/regex_e2e-965b23403f5f9fbd4d52b6bb4427638fee98b56c.md) | 648 |  4,090,656 |  207 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3036/ecrecover_e2e-965b23403f5f9fbd4d52b6bb4427638fee98b56c.md) | 220 |  112,210 |  180 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3036/pairing_e2e-965b23403f5f9fbd4d52b6bb4427638fee98b56c.md) | 292 |  592,827 |  174 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3036/kitchen_sink_e2e-965b23403f5f9fbd4d52b6bb4427638fee98b56c.md) | 2,293 |  1,979,971 |  457 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/965b23403f5f9fbd4d52b6bb4427638fee98b56c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29609763955)
