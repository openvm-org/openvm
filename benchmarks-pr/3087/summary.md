| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/fibonacci-11b5b11c4a9d8ca47100d2ca8dc506178187354a.md) | 458 |  4,000,051 |  240 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/keccak-11b5b11c4a9d8ca47100d2ca8dc506178187354a.md) | 7,235 |  14,365,133 |  1,540 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/sha2_bench-11b5b11c4a9d8ca47100d2ca8dc506178187354a.md) | 4,674 |  11,167,961 |  530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/regex-11b5b11c4a9d8ca47100d2ca8dc506178187354a.md) | 651 |  4,090,656 |  220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/ecrecover-11b5b11c4a9d8ca47100d2ca8dc506178187354a.md) | 222 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/pairing-11b5b11c4a9d8ca47100d2ca8dc506178187354a.md) | 305 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3087/kitchen_sink-11b5b11c4a9d8ca47100d2ca8dc506178187354a.md) | 2,758 |  1,979,971 |  468 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/11b5b11c4a9d8ca47100d2ca8dc506178187354a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30662402043)
