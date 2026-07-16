| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-77fa506781bc7b66d0b5b6791158956a71b938e7.md) | 471 |  4,000,051 |  238 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-77fa506781bc7b66d0b5b6791158956a71b938e7.md) | 7,091 |  14,365,133 |  1,515 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-77fa506781bc7b66d0b5b6791158956a71b938e7.md) | 4,474 |  11,167,961 |  530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-77fa506781bc7b66d0b5b6791158956a71b938e7.md) | 674 |  4,090,656 |  220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-77fa506781bc7b66d0b5b6791158956a71b938e7.md) | 219 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-77fa506781bc7b66d0b5b6791158956a71b938e7.md) | 253 |  592,827 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-77fa506781bc7b66d0b5b6791158956a71b938e7.md) | 2,712 |  1,979,971 |  471 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/77fa506781bc7b66d0b5b6791158956a71b938e7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29465260497)
