| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/fibonacci-d23f83d45f3958287ffe9e1bc79bfb35c82d2564.md) | 1,571 |  12,000,265 |  359 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/keccak-d23f83d45f3958287ffe9e1bc79bfb35c82d2564.md) | 9,326 |  18,655,329 |  1,523 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/sha2_bench-d23f83d45f3958287ffe9e1bc79bfb35c82d2564.md) | 4,859 |  14,793,960 |  572 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/regex-d23f83d45f3958287ffe9e1bc79bfb35c82d2564.md) | 665 |  4,137,067 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/ecrecover-d23f83d45f3958287ffe9e1bc79bfb35c82d2564.md) | 427 |  123,583 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/pairing-d23f83d45f3958287ffe9e1bc79bfb35c82d2564.md) | 570 |  1,745,757 |  191 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/kitchen_sink-d23f83d45f3958287ffe9e1bc79bfb35c82d2564.md) | 2,210 |  2,579,903 |  479 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d23f83d45f3958287ffe9e1bc79bfb35c82d2564

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31186921220)
