| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2894/fibonacci-9ea62f753d9fa9b84b768a6019299457207fff7f.md) | 3,079 |  12,000,265 |  681 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2894/keccak-9ea62f753d9fa9b84b768a6019299457207fff7f.md) | 16,745 |  18,655,329 |  3,112 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2894/sha2_bench-9ea62f753d9fa9b84b768a6019299457207fff7f.md) | 9,241 |  14,793,960 |  1,127 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2894/regex-9ea62f753d9fa9b84b768a6019299457207fff7f.md) | 1,166 |  4,137,067 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2894/ecrecover-9ea62f753d9fa9b84b768a6019299457207fff7f.md) | 599 |  123,583 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2894/pairing-9ea62f753d9fa9b84b768a6019299457207fff7f.md) | 935 |  1,745,757 |  305 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2894/kitchen_sink-9ea62f753d9fa9b84b768a6019299457207fff7f.md) | 4,131 |  2,579,903 |  884 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9ea62f753d9fa9b84b768a6019299457207fff7f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27645305031)
