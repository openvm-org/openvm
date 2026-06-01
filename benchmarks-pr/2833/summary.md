| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/fibonacci-3e93610f7a2a1f4862b148cf94940478bbce7428.md) | 3,694 |  12,000,265 |  905 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/keccak-3e93610f7a2a1f4862b148cf94940478bbce7428.md) | 18,964 |  18,655,329 |  3,347 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/sha2_bench-3e93610f7a2a1f4862b148cf94940478bbce7428.md) | 10,254 |  14,793,960 |  1,481 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/regex-3e93610f7a2a1f4862b148cf94940478bbce7428.md) | 1,424 |  4,137,067 |  363 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/ecrecover-3e93610f7a2a1f4862b148cf94940478bbce7428.md) | 600 |  123,583 |  252 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/pairing-3e93610f7a2a1f4862b148cf94940478bbce7428.md) | 885 |  1,745,757 |  267 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/kitchen_sink-3e93610f7a2a1f4862b148cf94940478bbce7428.md) | 1,890 |  2,579,903 |  409 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3e93610f7a2a1f4862b148cf94940478bbce7428

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26773308064)
