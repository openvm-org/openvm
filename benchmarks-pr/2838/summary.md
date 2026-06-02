| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2838/fibonacci-0fca90b873157892a18f164db7c73b48744ae2f9.md) | 3,762 |  12,000,265 |  918 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2838/keccak-0fca90b873157892a18f164db7c73b48744ae2f9.md) | 18,465 |  18,655,329 |  3,250 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2838/sha2_bench-0fca90b873157892a18f164db7c73b48744ae2f9.md) | 10,273 |  14,793,960 |  1,480 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2838/regex-0fca90b873157892a18f164db7c73b48744ae2f9.md) | 1,380 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2838/ecrecover-0fca90b873157892a18f164db7c73b48744ae2f9.md) | 600 |  123,583 |  251 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2838/pairing-0fca90b873157892a18f164db7c73b48744ae2f9.md) | 885 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2838/kitchen_sink-0fca90b873157892a18f164db7c73b48744ae2f9.md) | 1,886 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0fca90b873157892a18f164db7c73b48744ae2f9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26841819791)
