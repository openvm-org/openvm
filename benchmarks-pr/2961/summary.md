| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2961/fibonacci-f503b9ae7dff59dd01bdc94ff30f5f3a083958ce.md) | 3,063 |  12,000,265 |  671 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2961/keccak-f503b9ae7dff59dd01bdc94ff30f5f3a083958ce.md) | 16,331 |  18,655,329 |  3,046 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2961/sha2_bench-f503b9ae7dff59dd01bdc94ff30f5f3a083958ce.md) | 9,350 |  14,793,960 |  1,128 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2961/regex-f503b9ae7dff59dd01bdc94ff30f5f3a083958ce.md) | 1,179 |  4,137,067 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2961/ecrecover-f503b9ae7dff59dd01bdc94ff30f5f3a083958ce.md) | 605 |  123,583 |  293 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2961/pairing-f503b9ae7dff59dd01bdc94ff30f5f3a083958ce.md) | 933 |  1,745,757 |  305 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2961/kitchen_sink-f503b9ae7dff59dd01bdc94ff30f5f3a083958ce.md) | 4,138 |  2,579,903 |  896 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f503b9ae7dff59dd01bdc94ff30f5f3a083958ce

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28548015905)
