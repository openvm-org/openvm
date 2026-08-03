| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/fibonacci-c0d73280a35a82fe4b18b0f27688c0c72b04cbde.md) | 478 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/keccak-c0d73280a35a82fe4b18b0f27688c0c72b04cbde.md) | 7,381 |  14,365,133 |  1,521 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/sha2_bench-c0d73280a35a82fe4b18b0f27688c0c72b04cbde.md) | 4,164 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/regex-c0d73280a35a82fe4b18b0f27688c0c72b04cbde.md) | 660 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/ecrecover-c0d73280a35a82fe4b18b0f27688c0c72b04cbde.md) | 227 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/pairing-c0d73280a35a82fe4b18b0f27688c0c72b04cbde.md) | 240 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/kitchen_sink-c0d73280a35a82fe4b18b0f27688c0c72b04cbde.md) | 2,047 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c0d73280a35a82fe4b18b0f27688c0c72b04cbde

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30833928329)
