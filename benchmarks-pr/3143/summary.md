| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/fibonacci-52e0ff2d91419d7455357c9714dd3f3b702bb6e2.md) | 1,667 |  12,000,265 | <span style='color: green'>(-1 [-0.3%])</span> 367 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/keccak-52e0ff2d91419d7455357c9714dd3f3b702bb6e2.md) |<span style='color: green'>(-135 [-1.4%])</span> 9,504 |  18,655,329 | <span style='color: green'>(-17 [-1.1%])</span> 1,534 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/sha2_bench-52e0ff2d91419d7455357c9714dd3f3b702bb6e2.md) |<span style='color: red'>(+52 [+1.0%])</span> 5,328 |  14,793,960 | <span style='color: green'>(-3 [-0.5%])</span> 593 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/regex-52e0ff2d91419d7455357c9714dd3f3b702bb6e2.md) |<span style='color: red'>(+13 [+1.9%])</span> 699 |  4,137,067 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/ecrecover-52e0ff2d91419d7455357c9714dd3f3b702bb6e2.md) |<span style='color: green'>(-17 [-3.8%])</span> 430 |  123,583 | <span style='color: green'>(-2 [-1.0%])</span> 190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/pairing-52e0ff2d91419d7455357c9714dd3f3b702bb6e2.md) |<span style='color: green'>(-14 [-2.4%])</span> 569 |  1,745,757 | <span style='color: green'>(-4 [-2.0%])</span> 194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/kitchen_sink-52e0ff2d91419d7455357c9714dd3f3b702bb6e2.md) |<span style='color: green'>(-13 [-0.6%])</span> 2,290 |  2,579,903 | <span style='color: green'>(-5 [-1.0%])</span> 491 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/52e0ff2d91419d7455357c9714dd3f3b702bb6e2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33528750221)
