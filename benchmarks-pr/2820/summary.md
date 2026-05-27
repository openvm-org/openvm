| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2820/fibonacci-044485fee24463961b244fa40783f717be7acd5a.md) | 3,876 |  12,000,265 |  937 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2820/keccak-044485fee24463961b244fa40783f717be7acd5a.md) | 18,540 |  18,655,329 |  3,267 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2820/sha2_bench-044485fee24463961b244fa40783f717be7acd5a.md) | 10,154 |  14,793,960 |  1,452 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2820/regex-044485fee24463961b244fa40783f717be7acd5a.md) | 1,412 |  4,137,067 |  362 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2820/ecrecover-044485fee24463961b244fa40783f717be7acd5a.md) | 603 |  123,583 |  253 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2820/pairing-044485fee24463961b244fa40783f717be7acd5a.md) | 894 |  1,745,757 |  266 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2820/kitchen_sink-044485fee24463961b244fa40783f717be7acd5a.md) | 1,890 |  2,579,903 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/044485fee24463961b244fa40783f717be7acd5a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26521218578)
