| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/fibonacci-f46973a7d315848a788221754dde6dc2d103b56b.md) | 3,847 |  12,000,265 |  954 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/keccak-f46973a7d315848a788221754dde6dc2d103b56b.md) | 18,807 |  18,655,329 |  3,345 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/sha2_bench-f46973a7d315848a788221754dde6dc2d103b56b.md) | 9,047 |  14,793,960 |  1,401 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/regex-f46973a7d315848a788221754dde6dc2d103b56b.md) | 1,423 |  4,137,067 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/ecrecover-f46973a7d315848a788221754dde6dc2d103b56b.md) | 649 |  123,583 |  276 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/pairing-f46973a7d315848a788221754dde6dc2d103b56b.md) | 912 |  1,745,757 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/kitchen_sink-f46973a7d315848a788221754dde6dc2d103b56b.md) | 2,095 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f46973a7d315848a788221754dde6dc2d103b56b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24795811771)
