| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/fibonacci-607ac6baa1cd20a1c41b462a9f1749dc23b1f65c.md) | 3,766 |  12,000,265 |  916 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/keccak-607ac6baa1cd20a1c41b462a9f1749dc23b1f65c.md) | 18,541 |  18,655,329 |  3,260 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/sha2_bench-607ac6baa1cd20a1c41b462a9f1749dc23b1f65c.md) | 10,162 |  14,793,960 |  1,459 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/regex-607ac6baa1cd20a1c41b462a9f1749dc23b1f65c.md) | 1,403 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/ecrecover-607ac6baa1cd20a1c41b462a9f1749dc23b1f65c.md) | 624 |  123,583 |  250 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/pairing-607ac6baa1cd20a1c41b462a9f1749dc23b1f65c.md) | 897 |  1,745,757 |  265 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/kitchen_sink-607ac6baa1cd20a1c41b462a9f1749dc23b1f65c.md) | 1,903 |  2,579,903 |  414 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/607ac6baa1cd20a1c41b462a9f1749dc23b1f65c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26780816413)
