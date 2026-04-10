| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/fibonacci-a989023698f381383bac34f3d796c38296522326.md) | 3,846 |  12,000,265 |  961 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/keccak-a989023698f381383bac34f3d796c38296522326.md) | 19,025 |  18,655,329 |  3,391 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/regex-a989023698f381383bac34f3d796c38296522326.md) | 1,435 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/ecrecover-a989023698f381383bac34f3d796c38296522326.md) | 647 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/pairing-a989023698f381383bac34f3d796c38296522326.md) | 919 |  1,745,757 |  287 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/kitchen_sink-a989023698f381383bac34f3d796c38296522326.md) | 2,081 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a989023698f381383bac34f3d796c38296522326

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24264773917)
