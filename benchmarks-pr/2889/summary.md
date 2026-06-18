| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/fibonacci-461882f99fa66465a7dc47b78c2559c363e756ee.md) | 3,088 |  12,000,265 |  677 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/keccak-461882f99fa66465a7dc47b78c2559c363e756ee.md) | 16,352 |  18,655,329 |  3,036 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/sha2_bench-461882f99fa66465a7dc47b78c2559c363e756ee.md) | 9,382 |  14,793,960 |  1,144 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/regex-461882f99fa66465a7dc47b78c2559c363e756ee.md) | 1,175 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/ecrecover-461882f99fa66465a7dc47b78c2559c363e756ee.md) | 606 |  123,583 |  285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/pairing-461882f99fa66465a7dc47b78c2559c363e756ee.md) | 941 |  1,745,757 |  307 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/kitchen_sink-461882f99fa66465a7dc47b78c2559c363e756ee.md) | 4,111 |  2,579,903 |  880 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/461882f99fa66465a7dc47b78c2559c363e756ee

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27789003975)
