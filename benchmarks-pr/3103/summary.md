| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-e534fe7e8a04dc587640d81620efcc907adc0134.md) | 458 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-e534fe7e8a04dc587640d81620efcc907adc0134.md) | 7,352 |  14,365,133 |  1,525 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-e534fe7e8a04dc587640d81620efcc907adc0134.md) | 4,154 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-e534fe7e8a04dc587640d81620efcc907adc0134.md) | 662 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-e534fe7e8a04dc587640d81620efcc907adc0134.md) | 197 |  112,210 |  195 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-e534fe7e8a04dc587640d81620efcc907adc0134.md) | 239 |  592,827 |  195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-e534fe7e8a04dc587640d81620efcc907adc0134.md) | 2,038 |  1,979,971 |  524 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e534fe7e8a04dc587640d81620efcc907adc0134

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31813580616)
