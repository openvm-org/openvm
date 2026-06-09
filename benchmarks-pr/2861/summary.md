| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/fibonacci-d09e6a394e30202262ca211a75a2aa8217aceead.md) | 4,054 |  12,000,265 |  1,163 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/keccak-d09e6a394e30202262ca211a75a2aa8217aceead.md) | 21,815 |  18,655,329 |  4,618 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/sha2_bench-d09e6a394e30202262ca211a75a2aa8217aceead.md) | 9,625 |  14,793,960 |  1,842 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/regex-d09e6a394e30202262ca211a75a2aa8217aceead.md) | 1,513 |  4,137,067 |  431 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/ecrecover-d09e6a394e30202262ca211a75a2aa8217aceead.md) | 604 |  123,583 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/pairing-d09e6a394e30202262ca211a75a2aa8217aceead.md) | 935 |  1,745,757 |  309 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/kitchen_sink-d09e6a394e30202262ca211a75a2aa8217aceead.md) | 4,135 |  2,579,903 |  881 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d09e6a394e30202262ca211a75a2aa8217aceead

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27190247686)
