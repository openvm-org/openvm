| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2906/fibonacci-1008c1d1f013f926a7e147ee37dad9aebeac6cde.md) | 3,043 |  12,000,265 |  676 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2906/keccak-1008c1d1f013f926a7e147ee37dad9aebeac6cde.md) | 16,469 |  18,655,329 |  3,050 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2906/sha2_bench-1008c1d1f013f926a7e147ee37dad9aebeac6cde.md) | 9,152 |  14,793,960 |  1,126 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2906/regex-1008c1d1f013f926a7e147ee37dad9aebeac6cde.md) | 1,155 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2906/ecrecover-1008c1d1f013f926a7e147ee37dad9aebeac6cde.md) | 598 |  123,583 |  285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2906/pairing-1008c1d1f013f926a7e147ee37dad9aebeac6cde.md) | 943 |  1,745,757 |  312 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2906/kitchen_sink-1008c1d1f013f926a7e147ee37dad9aebeac6cde.md) | 4,133 |  2,579,903 |  892 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1008c1d1f013f926a7e147ee37dad9aebeac6cde

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27788853059)
