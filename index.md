| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-a20a9f903d83b6af54adc6fcdcfe12bf1a9df8ae.md) | 1,569 |  12,000,265 |  360 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-a20a9f903d83b6af54adc6fcdcfe12bf1a9df8ae.md) | 9,347 |  18,655,329 |  1,517 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-a20a9f903d83b6af54adc6fcdcfe12bf1a9df8ae.md) | 4,936 |  14,793,960 |  577 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-a20a9f903d83b6af54adc6fcdcfe12bf1a9df8ae.md) | 667 |  4,137,067 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-a20a9f903d83b6af54adc6fcdcfe12bf1a9df8ae.md) | 432 |  123,583 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-a20a9f903d83b6af54adc6fcdcfe12bf1a9df8ae.md) | 587 |  1,745,757 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-a20a9f903d83b6af54adc6fcdcfe12bf1a9df8ae.md) | 2,208 |  2,579,903 |  474 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a20a9f903d83b6af54adc6fcdcfe12bf1a9df8ae

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31405958772)
