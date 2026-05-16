| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2789/fibonacci-8eaa05e9da4008118c0628defd7c3fce9a6657fa.md) | 3,743 |  12,000,265 |  917 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2789/keccak-8eaa05e9da4008118c0628defd7c3fce9a6657fa.md) | 18,770 |  18,655,329 |  3,315 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2789/sha2_bench-8eaa05e9da4008118c0628defd7c3fce9a6657fa.md) | 10,062 |  14,793,960 |  1,437 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2789/regex-8eaa05e9da4008118c0628defd7c3fce9a6657fa.md) | 1,392 |  4,137,067 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2789/ecrecover-8eaa05e9da4008118c0628defd7c3fce9a6657fa.md) | 594 |  123,583 |  246 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2789/pairing-8eaa05e9da4008118c0628defd7c3fce9a6657fa.md) | 895 |  1,745,757 |  263 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2789/kitchen_sink-8eaa05e9da4008118c0628defd7c3fce9a6657fa.md) | 1,889 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8eaa05e9da4008118c0628defd7c3fce9a6657fa

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25959144280)
