| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2875/fibonacci-b55316266c4d465b8a3370019bf74fbadcc1311e.md) | 3,960 |  12,000,265 |  1,144 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2875/keccak-b55316266c4d465b8a3370019bf74fbadcc1311e.md) | 21,920 |  18,655,329 |  4,653 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2875/sha2_bench-b55316266c4d465b8a3370019bf74fbadcc1311e.md) | 9,581 |  14,793,960 |  1,831 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2875/regex-b55316266c4d465b8a3370019bf74fbadcc1311e.md) | 1,504 |  4,137,067 |  430 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2875/ecrecover-b55316266c4d465b8a3370019bf74fbadcc1311e.md) | 604 |  123,583 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2875/pairing-b55316266c4d465b8a3370019bf74fbadcc1311e.md) | 944 |  1,745,757 |  301 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2875/kitchen_sink-b55316266c4d465b8a3370019bf74fbadcc1311e.md) | 4,125 |  2,579,903 |  880 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b55316266c4d465b8a3370019bf74fbadcc1311e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27348333487)
