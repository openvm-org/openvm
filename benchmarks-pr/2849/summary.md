| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2849/fibonacci-d3f4314ba20c3bb071f64d7d5b37f23b633f46a9.md) | 3,757 |  12,000,265 |  923 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2849/keccak-d3f4314ba20c3bb071f64d7d5b37f23b633f46a9.md) | 18,063 |  18,655,329 |  3,286 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2849/sha2_bench-d3f4314ba20c3bb071f64d7d5b37f23b633f46a9.md) | 9,960 |  14,793,960 |  1,447 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2849/regex-d3f4314ba20c3bb071f64d7d5b37f23b633f46a9.md) | 1,397 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2849/ecrecover-d3f4314ba20c3bb071f64d7d5b37f23b633f46a9.md) | 603 |  123,583 |  252 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2849/pairing-d3f4314ba20c3bb071f64d7d5b37f23b633f46a9.md) | 878 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2849/kitchen_sink-d3f4314ba20c3bb071f64d7d5b37f23b633f46a9.md) | 3,906 |  2,579,903 |  968 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d3f4314ba20c3bb071f64d7d5b37f23b633f46a9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27045404074)
