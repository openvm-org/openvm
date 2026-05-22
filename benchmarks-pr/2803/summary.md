| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/fibonacci-7a7aaac0a3e656d7f8e5e250ab7fee81375237d3.md) | 3,844 |  12,000,265 |  928 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/keccak-7a7aaac0a3e656d7f8e5e250ab7fee81375237d3.md) | 18,735 |  18,655,329 |  3,295 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/sha2_bench-7a7aaac0a3e656d7f8e5e250ab7fee81375237d3.md) | 10,054 |  14,793,960 |  1,443 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/regex-7a7aaac0a3e656d7f8e5e250ab7fee81375237d3.md) | 1,387 |  4,137,067 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/ecrecover-7a7aaac0a3e656d7f8e5e250ab7fee81375237d3.md) | 601 |  123,583 |  246 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/pairing-7a7aaac0a3e656d7f8e5e250ab7fee81375237d3.md) | 878 |  1,745,757 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/kitchen_sink-7a7aaac0a3e656d7f8e5e250ab7fee81375237d3.md) | 1,899 |  2,579,903 |  408 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7a7aaac0a3e656d7f8e5e250ab7fee81375237d3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26260411555)
