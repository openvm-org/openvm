| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2782/fibonacci-7ae29ae48cddb3ad36ae9ffbad52ebdbdf94825f.md) | 3,739 |  12,000,265 |  915 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2782/keccak-7ae29ae48cddb3ad36ae9ffbad52ebdbdf94825f.md) | 18,899 |  18,655,329 |  3,326 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2782/sha2_bench-7ae29ae48cddb3ad36ae9ffbad52ebdbdf94825f.md) | 10,114 |  14,793,960 |  1,458 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2782/regex-7ae29ae48cddb3ad36ae9ffbad52ebdbdf94825f.md) | 1,409 |  4,137,067 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2782/ecrecover-7ae29ae48cddb3ad36ae9ffbad52ebdbdf94825f.md) | 598 |  123,583 |  250 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2782/pairing-7ae29ae48cddb3ad36ae9ffbad52ebdbdf94825f.md) | 894 |  1,745,757 |  264 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2782/kitchen_sink-7ae29ae48cddb3ad36ae9ffbad52ebdbdf94825f.md) | 1,897 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7ae29ae48cddb3ad36ae9ffbad52ebdbdf94825f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25934924488)
