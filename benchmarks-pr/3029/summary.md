| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/fibonacci-28ed08b61696f282f82d00dca44d111518d22e6e.md) |<span style='color: green'>(-10 [-0.6%])</span> 1,576 |  12,000,265 | <span style='color: green'>(-2 [-0.6%])</span> 359 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/keccak-28ed08b61696f282f82d00dca44d111518d22e6e.md) |<span style='color: red'>(+129 [+1.4%])</span> 9,384 |  18,655,329 | <span style='color: red'>(+28 [+1.8%])</span> 1,543 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/sha2_bench-28ed08b61696f282f82d00dca44d111518d22e6e.md) |<span style='color: red'>(+21 [+0.4%])</span> 4,896 |  14,793,960 | <span style='color: red'>(+4 [+0.7%])</span> 576 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/regex-28ed08b61696f282f82d00dca44d111518d22e6e.md) |<span style='color: red'>(+3 [+0.5%])</span> 665 |  4,137,067 | <span style='color: red'>(+3 [+1.4%])</span> 213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/ecrecover-28ed08b61696f282f82d00dca44d111518d22e6e.md) |<span style='color: red'>(+6 [+1.4%])</span> 433 |  123,583 | <span style='color: red'>(+1 [+0.5%])</span> 186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/pairing-28ed08b61696f282f82d00dca44d111518d22e6e.md) |<span style='color: green'>(-7 [-1.2%])</span> 563 |  1,745,757 | <span style='color: green'>(-1 [-0.5%])</span> 191 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3029/kitchen_sink-28ed08b61696f282f82d00dca44d111518d22e6e.md) |<span style='color: green'>(-12 [-0.5%])</span> 2,202 |  2,579,903 | <span style='color: red'>(+9 [+1.9%])</span> 486 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/28ed08b61696f282f82d00dca44d111518d22e6e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30124946222)
