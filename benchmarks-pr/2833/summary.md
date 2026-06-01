| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/fibonacci-87db76e9ddf89c99a90f267015be72da63269a1b.md) | 3,812 |  12,000,265 |  922 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/keccak-87db76e9ddf89c99a90f267015be72da63269a1b.md) | 18,656 |  18,655,329 |  3,291 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/sha2_bench-87db76e9ddf89c99a90f267015be72da63269a1b.md) | 10,343 |  14,793,960 |  1,486 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/regex-87db76e9ddf89c99a90f267015be72da63269a1b.md) | 1,382 |  4,137,067 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/ecrecover-87db76e9ddf89c99a90f267015be72da63269a1b.md) | 598 |  123,583 |  245 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/pairing-87db76e9ddf89c99a90f267015be72da63269a1b.md) | 893 |  1,745,757 |  261 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/kitchen_sink-87db76e9ddf89c99a90f267015be72da63269a1b.md) | 1,900 |  2,579,903 |  411 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/87db76e9ddf89c99a90f267015be72da63269a1b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26774474496)
