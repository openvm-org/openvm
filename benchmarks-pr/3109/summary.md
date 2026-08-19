| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/fibonacci-b0025d0c0bc5f9b118f27d4de80b47677d5dcce2.md) | 467 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/keccak-b0025d0c0bc5f9b118f27d4de80b47677d5dcce2.md) | 7,421 |  14,365,133 |  1,531 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/sha2_bench-b0025d0c0bc5f9b118f27d4de80b47677d5dcce2.md) | 4,288 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/regex-b0025d0c0bc5f9b118f27d4de80b47677d5dcce2.md) | 687 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/ecrecover-b0025d0c0bc5f9b118f27d4de80b47677d5dcce2.md) | 196 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/pairing-b0025d0c0bc5f9b118f27d4de80b47677d5dcce2.md) | 229 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/kitchen_sink-b0025d0c0bc5f9b118f27d4de80b47677d5dcce2.md) | 2,044 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b0025d0c0bc5f9b118f27d4de80b47677d5dcce2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32285218817)
