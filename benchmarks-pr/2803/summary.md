| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/fibonacci-bc449d4a08299e16d01fac1295cc822f5cf87cae.md) | 3,749 |  12,000,265 |  910 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/keccak-bc449d4a08299e16d01fac1295cc822f5cf87cae.md) | 18,606 |  18,655,329 |  3,280 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/sha2_bench-bc449d4a08299e16d01fac1295cc822f5cf87cae.md) | 9,965 |  14,793,960 |  1,432 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/regex-bc449d4a08299e16d01fac1295cc822f5cf87cae.md) | 1,401 |  4,137,067 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/ecrecover-bc449d4a08299e16d01fac1295cc822f5cf87cae.md) | 598 |  123,583 |  249 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/pairing-bc449d4a08299e16d01fac1295cc822f5cf87cae.md) | 882 |  1,745,757 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/kitchen_sink-bc449d4a08299e16d01fac1295cc822f5cf87cae.md) | 1,912 |  2,579,903 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bc449d4a08299e16d01fac1295cc822f5cf87cae

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26307399326)
