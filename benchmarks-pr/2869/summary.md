| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2869/fibonacci-7719b49dee55f248ed32d9cbb1904a3c791752b2.md) | 4,061 |  12,000,265 |  1,161 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2869/keccak-7719b49dee55f248ed32d9cbb1904a3c791752b2.md) | 21,638 |  18,655,329 |  4,579 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2869/sha2_bench-7719b49dee55f248ed32d9cbb1904a3c791752b2.md) | 9,745 |  14,793,960 |  1,862 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2869/regex-7719b49dee55f248ed32d9cbb1904a3c791752b2.md) | 1,503 |  4,137,067 |  430 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2869/ecrecover-7719b49dee55f248ed32d9cbb1904a3c791752b2.md) | 610 |  123,583 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2869/pairing-7719b49dee55f248ed32d9cbb1904a3c791752b2.md) | 941 |  1,745,757 |  312 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2869/kitchen_sink-7719b49dee55f248ed32d9cbb1904a3c791752b2.md) | 4,140 |  2,579,903 |  884 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7719b49dee55f248ed32d9cbb1904a3c791752b2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27311733985)
