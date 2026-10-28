| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3180/fibonacci-7527814f898023f156dd14776a134b2bd7697ee7.md) |<span style='color: red'>(+10 [+0.6%])</span> 1,696 |  12,000,265 | <span style='color: red'>(+4 [+1.1%])</span> 371 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3180/keccak-7527814f898023f156dd14776a134b2bd7697ee7.md) |<span style='color: red'>(+14 [+0.1%])</span> 9,688 |  18,655,329 |  1,554 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3180/sha2_bench-7527814f898023f156dd14776a134b2bd7697ee7.md) | 5,337 |  14,793,960 |  592 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3180/regex-7527814f898023f156dd14776a134b2bd7697ee7.md) |<span style='color: red'>(+2 [+0.3%])</span> 706 |  4,137,067 | <span style='color: red'>(+1 [+0.5%])</span> 221 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3180/ecrecover-7527814f898023f156dd14776a134b2bd7697ee7.md) |<span style='color: red'>(+9 [+2.1%])</span> 444 |  123,583 | <span style='color: red'>(+1 [+0.5%])</span> 190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3180/pairing-7527814f898023f156dd14776a134b2bd7697ee7.md) |<span style='color: green'>(-40 [-6.6%])</span> 566 |  1,745,757 |  194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3180/kitchen_sink-7527814f898023f156dd14776a134b2bd7697ee7.md) |<span style='color: red'>(+3 [+0.1%])</span> 2,291 |  2,579,903 | <span style='color: red'>(+1 [+0.2%])</span> 495 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7527814f898023f156dd14776a134b2bd7697ee7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/40709461384)
