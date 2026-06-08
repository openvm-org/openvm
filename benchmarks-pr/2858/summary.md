| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2858/fibonacci-8722ae8e09298905050bb7333d5ad2ffbde29468.md) | 3,736 |  12,000,265 |  919 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2858/keccak-8722ae8e09298905050bb7333d5ad2ffbde29468.md) | 18,190 |  18,655,329 |  3,304 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2858/sha2_bench-8722ae8e09298905050bb7333d5ad2ffbde29468.md) | 9,958 |  14,793,960 |  1,452 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2858/regex-8722ae8e09298905050bb7333d5ad2ffbde29468.md) | 1,398 |  4,137,067 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2858/ecrecover-8722ae8e09298905050bb7333d5ad2ffbde29468.md) | 596 |  123,583 |  253 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2858/pairing-8722ae8e09298905050bb7333d5ad2ffbde29468.md) | 876 |  1,745,757 |  259 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2858/kitchen_sink-8722ae8e09298905050bb7333d5ad2ffbde29468.md) | 3,817 |  2,579,903 |  945 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8722ae8e09298905050bb7333d5ad2ffbde29468

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27153030956)
