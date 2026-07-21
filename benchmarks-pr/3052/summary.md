| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/fibonacci-c7703e214419d25c61385401b457203fda3facd3.md) |<span style='color: green'>(-9 [-0.6%])</span> 1,577 |  12,000,265 | <span style='color: green'>(-2 [-0.6%])</span> 359 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/keccak-c7703e214419d25c61385401b457203fda3facd3.md) |<span style='color: green'>(-45 [-0.5%])</span> 9,210 |  18,655,329 | <span style='color: green'>(-17 [-1.1%])</span> 1,498 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/sha2_bench-c7703e214419d25c61385401b457203fda3facd3.md) |<span style='color: red'>(+135 [+2.8%])</span> 5,010 |  14,793,960 | <span style='color: red'>(+8 [+1.4%])</span> 580 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/regex-c7703e214419d25c61385401b457203fda3facd3.md) |<span style='color: red'>(+3 [+0.5%])</span> 665 |  4,137,067 | <span style='color: red'>(+3 [+1.4%])</span> 213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/ecrecover-c7703e214419d25c61385401b457203fda3facd3.md) |<span style='color: red'>(+10 [+2.3%])</span> 437 |  123,583 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/pairing-c7703e214419d25c61385401b457203fda3facd3.md) |<span style='color: green'>(-17 [-3.0%])</span> 553 |  1,745,757 | <span style='color: green'>(-6 [-3.1%])</span> 186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3052/kitchen_sink-c7703e214419d25c61385401b457203fda3facd3.md) |<span style='color: green'>(-4 [-0.2%])</span> 2,210 |  2,579,903 |  477 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c7703e214419d25c61385401b457203fda3facd3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29839866962)
