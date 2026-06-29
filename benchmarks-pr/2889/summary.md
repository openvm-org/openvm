| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/fibonacci-14cac7ee0097bf94a522d4daea38812418035620.md) | 3,049 |  12,000,265 |  674 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/keccak-14cac7ee0097bf94a522d4daea38812418035620.md) | 16,744 |  18,655,329 |  3,105 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/sha2_bench-14cac7ee0097bf94a522d4daea38812418035620.md) | 9,185 |  14,793,960 |  1,129 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/regex-14cac7ee0097bf94a522d4daea38812418035620.md) | 1,165 |  4,137,067 |  350 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/ecrecover-14cac7ee0097bf94a522d4daea38812418035620.md) | 605 |  123,583 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/pairing-14cac7ee0097bf94a522d4daea38812418035620.md) | 952 |  1,745,757 |  315 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/kitchen_sink-14cac7ee0097bf94a522d4daea38812418035620.md) | 4,115 |  2,579,903 |  882 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/14cac7ee0097bf94a522d4daea38812418035620

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28379362176)
