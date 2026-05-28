| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2825/fibonacci-7bd82e0564e4bdf2760f25cefba826942ff03316.md) | 3,832 |  12,000,265 |  932 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2825/keccak-7bd82e0564e4bdf2760f25cefba826942ff03316.md) | 18,221 |  18,655,329 |  3,310 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2825/sha2_bench-7bd82e0564e4bdf2760f25cefba826942ff03316.md) | 10,291 |  14,793,960 |  1,486 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2825/regex-7bd82e0564e4bdf2760f25cefba826942ff03316.md) | 1,393 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2825/ecrecover-7bd82e0564e4bdf2760f25cefba826942ff03316.md) | 613 |  123,583 |  260 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2825/pairing-7bd82e0564e4bdf2760f25cefba826942ff03316.md) | 898 |  1,745,757 |  267 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2825/kitchen_sink-7bd82e0564e4bdf2760f25cefba826942ff03316.md) | 1,871 |  2,579,903 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7bd82e0564e4bdf2760f25cefba826942ff03316

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26600128882)
