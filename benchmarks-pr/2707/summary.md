| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2707/fibonacci-def1d149919141f15067f3819968e233a3579524.md) | 3,808 |  12,000,265 |  941 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2707/keccak-def1d149919141f15067f3819968e233a3579524.md) | 18,723 |  18,655,329 |  3,335 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2707/sha2_bench-def1d149919141f15067f3819968e233a3579524.md) | 10,023 |  14,793,960 |  1,419 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2707/regex-def1d149919141f15067f3819968e233a3579524.md) | 1,424 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2707/ecrecover-def1d149919141f15067f3819968e233a3579524.md) | 649 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2707/pairing-def1d149919141f15067f3819968e233a3579524.md) | 907 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2707/kitchen_sink-def1d149919141f15067f3819968e233a3579524.md) | 2,153 |  2,579,903 |  433 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/def1d149919141f15067f3819968e233a3579524

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24462151879)
