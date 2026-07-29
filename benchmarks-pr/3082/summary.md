| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3082/fibonacci-c1832cfb677c902666f384c2f9bf770cf098913a.md) | 458 |  4,000,051 |  240 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3082/keccak-c1832cfb677c902666f384c2f9bf770cf098913a.md) | 7,223 |  14,365,133 |  1,569 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3082/sha2_bench-c1832cfb677c902666f384c2f9bf770cf098913a.md) | 4,686 |  11,167,961 |  528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3082/regex-c1832cfb677c902666f384c2f9bf770cf098913a.md) | 654 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3082/ecrecover-c1832cfb677c902666f384c2f9bf770cf098913a.md) | 229 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3082/pairing-c1832cfb677c902666f384c2f9bf770cf098913a.md) | 299 |  592,827 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3082/kitchen_sink-c1832cfb677c902666f384c2f9bf770cf098913a.md) | 2,645 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c1832cfb677c902666f384c2f9bf770cf098913a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30437148623)
