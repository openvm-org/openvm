| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3132/fibonacci-052fac775e6410b56a39e6e87d1d609969568a74.md) |<span style='color: green'>(-26 [-1.5%])</span> 1,658 |  12,000,265 | <span style='color: green'>(-1 [-0.3%])</span> 369 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3132/keccak-052fac775e6410b56a39e6e87d1d609969568a74.md) |<span style='color: green'>(-44 [-0.5%])</span> 9,474 |  18,655,329 | <span style='color: green'>(-28 [-1.8%])</span> 1,536 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3132/sha2_bench-052fac775e6410b56a39e6e87d1d609969568a74.md) |<span style='color: green'>(-109 [-2.0%])</span> 5,248 |  14,793,960 | <span style='color: red'>(+5 [+0.9%])</span> 591 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3132/regex-052fac775e6410b56a39e6e87d1d609969568a74.md) |<span style='color: red'>(+11 [+1.6%])</span> 708 |  4,137,067 | <span style='color: red'>(+3 [+1.4%])</span> 219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3132/ecrecover-052fac775e6410b56a39e6e87d1d609969568a74.md) |<span style='color: red'>(+10 [+2.3%])</span> 440 |  123,583 | <span style='color: green'>(-1 [-0.5%])</span> 188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3132/pairing-052fac775e6410b56a39e6e87d1d609969568a74.md) |<span style='color: red'>(+21 [+3.6%])</span> 611 |  1,745,757 | <span style='color: red'>(+2 [+1.0%])</span> 195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3132/kitchen_sink-052fac775e6410b56a39e6e87d1d609969568a74.md) |<span style='color: green'>(-4 [-0.2%])</span> 2,296 |  2,579,903 | <span style='color: red'>(+1 [+0.2%])</span> 493 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/052fac775e6410b56a39e6e87d1d609969568a74

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33075926858)
