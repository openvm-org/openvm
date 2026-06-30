| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2951/fibonacci-732b2e86d70c5334b18c622745d7590062acf557.md) | 3,074 |  12,000,265 |  678 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2951/keccak-732b2e86d70c5334b18c622745d7590062acf557.md) | 16,292 |  18,655,329 |  3,023 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2951/sha2_bench-732b2e86d70c5334b18c622745d7590062acf557.md) | 9,127 |  14,793,960 |  1,116 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2951/regex-732b2e86d70c5334b18c622745d7590062acf557.md) | 1,173 |  4,137,067 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2951/ecrecover-732b2e86d70c5334b18c622745d7590062acf557.md) | 604 |  123,583 |  286 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2951/pairing-732b2e86d70c5334b18c622745d7590062acf557.md) | 934 |  1,745,757 |  307 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2951/kitchen_sink-732b2e86d70c5334b18c622745d7590062acf557.md) | 4,184 |  2,579,903 |  903 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/732b2e86d70c5334b18c622745d7590062acf557

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28465309801)
