| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2875/fibonacci-16c5051b8804bf0f280b043b3f2140081109d66e.md) | 3,978 |  12,000,265 |  1,150 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2875/keccak-16c5051b8804bf0f280b043b3f2140081109d66e.md) | 22,055 |  18,655,329 |  4,702 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2875/sha2_bench-16c5051b8804bf0f280b043b3f2140081109d66e.md) | 9,602 |  14,793,960 |  1,835 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2875/regex-16c5051b8804bf0f280b043b3f2140081109d66e.md) | 1,495 |  4,137,067 |  425 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2875/ecrecover-16c5051b8804bf0f280b043b3f2140081109d66e.md) | 605 |  123,583 |  286 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2875/pairing-16c5051b8804bf0f280b043b3f2140081109d66e.md) | 949 |  1,745,757 |  306 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2875/kitchen_sink-16c5051b8804bf0f280b043b3f2140081109d66e.md) | 4,106 |  2,579,903 |  876 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/16c5051b8804bf0f280b043b3f2140081109d66e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27359215863)
