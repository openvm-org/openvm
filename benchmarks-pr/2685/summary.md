| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/fibonacci-ef847862f804858426bb4ac55f8c0151676befda.md) | 3,810 |  12,000,265 |  952 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/keccak-ef847862f804858426bb4ac55f8c0151676befda.md) | 18,910 |  18,655,329 |  3,402 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/regex-ef847862f804858426bb4ac55f8c0151676befda.md) | 1,417 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/ecrecover-ef847862f804858426bb4ac55f8c0151676befda.md) | 648 |  123,583 |  275 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/pairing-ef847862f804858426bb4ac55f8c0151676befda.md) | 909 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2685/kitchen_sink-ef847862f804858426bb4ac55f8c0151676befda.md) | 2,161 |  2,579,903 |  433 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ef847862f804858426bb4ac55f8c0151676befda

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24256983167)
