| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/fibonacci-14cc871cf5d4c9aec2a9a4eae020955e20258ba5.md) | 3,719 |  12,000,265 |  909 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/keccak-14cc871cf5d4c9aec2a9a4eae020955e20258ba5.md) | 18,406 |  18,655,329 |  3,249 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/sha2_bench-14cc871cf5d4c9aec2a9a4eae020955e20258ba5.md) | 10,219 |  14,793,960 |  1,460 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/regex-14cc871cf5d4c9aec2a9a4eae020955e20258ba5.md) | 1,405 |  4,137,067 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/ecrecover-14cc871cf5d4c9aec2a9a4eae020955e20258ba5.md) | 596 |  123,583 |  247 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/pairing-14cc871cf5d4c9aec2a9a4eae020955e20258ba5.md) | 892 |  1,745,757 |  264 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/kitchen_sink-14cc871cf5d4c9aec2a9a4eae020955e20258ba5.md) | 1,891 |  2,579,903 |  409 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/14cc871cf5d4c9aec2a9a4eae020955e20258ba5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26304258113)
