| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/fibonacci-6868f8e9efc8387d996a5d5048497619eeb8f4f3.md) | 413 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/keccak-6868f8e9efc8387d996a5d5048497619eeb8f4f3.md) | 8,556 |  14,365,133 |  1,517 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/sha2_bench-6868f8e9efc8387d996a5d5048497619eeb8f4f3.md) | 4,222 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/regex-6868f8e9efc8387d996a5d5048497619eeb8f4f3.md) | 570 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/ecrecover-6868f8e9efc8387d996a5d5048497619eeb8f4f3.md) | 219 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/pairing-6868f8e9efc8387d996a5d5048497619eeb8f4f3.md) | 287 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/kitchen_sink-6868f8e9efc8387d996a5d5048497619eeb8f4f3.md) | 2,043 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6868f8e9efc8387d996a5d5048497619eeb8f4f3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29563262577)
