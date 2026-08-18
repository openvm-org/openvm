| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-4926b40c1a431b777f3e040a84e0e398b990960c.md) | 461 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-4926b40c1a431b777f3e040a84e0e398b990960c.md) | 7,334 |  14,365,133 |  1,519 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-4926b40c1a431b777f3e040a84e0e398b990960c.md) | 4,157 |  11,167,961 |  528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-4926b40c1a431b777f3e040a84e0e398b990960c.md) | 668 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-4926b40c1a431b777f3e040a84e0e398b990960c.md) | 196 |  112,210 |  199 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-4926b40c1a431b777f3e040a84e0e398b990960c.md) | 237 |  592,827 |  196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-4926b40c1a431b777f3e040a84e0e398b990960c.md) | 2,033 |  1,979,971 |  527 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4926b40c1a431b777f3e040a84e0e398b990960c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32164356875)
