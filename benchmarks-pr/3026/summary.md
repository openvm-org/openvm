| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/fibonacci-9ad0da184976b132549f70abb888ac3acf7e7ea0.md) | 409 |  4,000,051 |  226 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/keccak-9ad0da184976b132549f70abb888ac3acf7e7ea0.md) | 8,466 |  14,365,133 |  1,536 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/sha2_bench-9ad0da184976b132549f70abb888ac3acf7e7ea0.md) | 3,944 |  11,167,961 |  527 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/regex-9ad0da184976b132549f70abb888ac3acf7e7ea0.md) | 571 |  4,090,656 |  211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/ecrecover-9ad0da184976b132549f70abb888ac3acf7e7ea0.md) | 217 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/pairing-9ad0da184976b132549f70abb888ac3acf7e7ea0.md) | 268 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/kitchen_sink-9ad0da184976b132549f70abb888ac3acf7e7ea0.md) | 1,914 |  1,979,971 |  468 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9ad0da184976b132549f70abb888ac3acf7e7ea0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29446305750)
