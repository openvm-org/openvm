| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/fibonacci-76132e80617c765c76472b464b9d100bd4cebc3f.md) | 3,799 |  12,000,265 |  948 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/keccak-76132e80617c765c76472b464b9d100bd4cebc3f.md) | 18,371 |  18,655,329 |  3,279 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/sha2_bench-76132e80617c765c76472b464b9d100bd4cebc3f.md) | 9,087 |  14,793,960 |  1,411 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/regex-76132e80617c765c76472b464b9d100bd4cebc3f.md) | 1,409 |  4,137,067 |  370 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/ecrecover-76132e80617c765c76472b464b9d100bd4cebc3f.md) | 646 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/pairing-76132e80617c765c76472b464b9d100bd4cebc3f.md) | 903 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/kitchen_sink-76132e80617c765c76472b464b9d100bd4cebc3f.md) | 2,086 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/76132e80617c765c76472b464b9d100bd4cebc3f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24597337571)
