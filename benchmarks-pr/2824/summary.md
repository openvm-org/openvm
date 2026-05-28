| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2824/fibonacci-7bd82e0564e4bdf2760f25cefba826942ff03316.md) | 3,832 |  12,000,265 | <span style='color: green'>(-3554 [-79.2%])</span> 932 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2824/keccak-7bd82e0564e4bdf2760f25cefba826942ff03316.md) | 18,221 |  18,655,329 |  3,310 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2824/sha2_bench-7bd82e0564e4bdf2760f25cefba826942ff03316.md) | 10,291 |  14,793,960 |  1,486 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2824/regex-7bd82e0564e4bdf2760f25cefba826942ff03316.md) | 1,393 |  4,137,067 | <span style='color: green'>(-11641 [-97.0%])</span> 356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2824/ecrecover-7bd82e0564e4bdf2760f25cefba826942ff03316.md) | 613 |  123,583 | <span style='color: green'>(-5596 [-95.6%])</span> 260 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2824/pairing-7bd82e0564e4bdf2760f25cefba826942ff03316.md) | 898 |  1,745,757 | <span style='color: green'>(-6113 [-95.8%])</span> 267 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2824/kitchen_sink-7bd82e0564e4bdf2760f25cefba826942ff03316.md) | 1,855 |  2,579,903 |  409 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7bd82e0564e4bdf2760f25cefba826942ff03316

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26600073209)
