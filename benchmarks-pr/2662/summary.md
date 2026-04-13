| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/fibonacci-36ad9449676296e5fba5192f5ffa5d4fb116cdaf.md) | 3,849 |  12,000,265 |  959 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/keccak-36ad9449676296e5fba5192f5ffa5d4fb116cdaf.md) | 18,626 |  18,655,329 |  3,319 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/sha2_bench-36ad9449676296e5fba5192f5ffa5d4fb116cdaf.md) | 9,847 |  14,793,960 |  1,393 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/regex-36ad9449676296e5fba5192f5ffa5d4fb116cdaf.md) | 1,418 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/ecrecover-36ad9449676296e5fba5192f5ffa5d4fb116cdaf.md) | 651 |  123,583 |  276 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/pairing-36ad9449676296e5fba5192f5ffa5d4fb116cdaf.md) | 907 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/kitchen_sink-36ad9449676296e5fba5192f5ffa5d4fb116cdaf.md) | 2,139 |  2,579,903 |  432 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/36ad9449676296e5fba5192f5ffa5d4fb116cdaf

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24354748524)
