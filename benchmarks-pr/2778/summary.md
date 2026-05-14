| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-112c84876de1375a1f260aea0c92d5edf6359725.md) | 1,584 |  4,000,051 |  453 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-112c84876de1375a1f260aea0c92d5edf6359725.md) | 13,818 |  14,365,133 |  2,386 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-112c84876de1375a1f260aea0c92d5edf6359725.md) | 9,200 |  11,167,961 |  1,404 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-112c84876de1375a1f260aea0c92d5edf6359725.md) | 1,488 |  4,090,656 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-112c84876de1375a1f260aea0c92d5edf6359725.md) | 513 |  112,210 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-112c84876de1375a1f260aea0c92d5edf6359725.md) | 610 |  592,827 |  279 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-112c84876de1375a1f260aea0c92d5edf6359725.md) | 1,941 |  1,979,971 |  431 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/112c84876de1375a1f260aea0c92d5edf6359725

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25878570456)
