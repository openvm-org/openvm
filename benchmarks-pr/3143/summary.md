| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/fibonacci-9826be6e65d34aeff4a4e4d456e1dc8a85acdc3b.md) |<span style='color: red'>(+3 [+0.2%])</span> 1,670 |  12,000,265 |  368 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/keccak-9826be6e65d34aeff4a4e4d456e1dc8a85acdc3b.md) |<span style='color: green'>(-144 [-1.5%])</span> 9,495 |  18,655,329 | <span style='color: green'>(-23 [-1.5%])</span> 1,528 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/sha2_bench-9826be6e65d34aeff4a4e4d456e1dc8a85acdc3b.md) |<span style='color: red'>(+15 [+0.3%])</span> 5,291 |  14,793,960 | <span style='color: green'>(-6 [-1.0%])</span> 590 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/regex-9826be6e65d34aeff4a4e4d456e1dc8a85acdc3b.md) |<span style='color: red'>(+5 [+0.7%])</span> 691 |  4,137,067 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/ecrecover-9826be6e65d34aeff4a4e4d456e1dc8a85acdc3b.md) |<span style='color: green'>(-12 [-2.7%])</span> 435 |  123,583 |  192 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/pairing-9826be6e65d34aeff4a4e4d456e1dc8a85acdc3b.md) |<span style='color: green'>(-6 [-1.0%])</span> 577 |  1,745,757 | <span style='color: green'>(-4 [-2.0%])</span> 194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3143/kitchen_sink-9826be6e65d34aeff4a4e4d456e1dc8a85acdc3b.md) |<span style='color: red'>(+15 [+0.7%])</span> 2,318 |  2,579,903 | <span style='color: red'>(+2 [+0.4%])</span> 498 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9826be6e65d34aeff4a4e4d456e1dc8a85acdc3b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33530484815)
