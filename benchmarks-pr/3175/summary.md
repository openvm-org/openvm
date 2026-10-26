| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3175/fibonacci-48d938a7433a6fd25390daebc632bebb8397c90b.md) |<span style='color: red'>(+4 [+0.2%])</span> 1,683 |  12,000,265 | <span style='color: green'>(-4 [-1.1%])</span> 367 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3175/keccak-48d938a7433a6fd25390daebc632bebb8397c90b.md) |<span style='color: green'>(-117 [-1.2%])</span> 9,533 |  18,655,329 | <span style='color: green'>(-29 [-1.9%])</span> 1,528 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3175/sha2_bench-48d938a7433a6fd25390daebc632bebb8397c90b.md) | 5,310 |  14,793,960 | <span style='color: green'>(-2 [-0.3%])</span> 590 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3175/regex-48d938a7433a6fd25390daebc632bebb8397c90b.md) |<span style='color: green'>(-3 [-0.4%])</span> 699 |  4,137,067 | <span style='color: red'>(+5 [+2.3%])</span> 221 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3175/ecrecover-48d938a7433a6fd25390daebc632bebb8397c90b.md) |<span style='color: green'>(-3 [-0.7%])</span> 440 |  123,583 | <span style='color: green'>(-3 [-1.6%])</span> 190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3175/pairing-48d938a7433a6fd25390daebc632bebb8397c90b.md) |<span style='color: green'>(-6 [-1.0%])</span> 576 |  1,745,757 | <span style='color: red'>(+1 [+0.5%])</span> 195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3175/kitchen_sink-48d938a7433a6fd25390daebc632bebb8397c90b.md) |<span style='color: green'>(-9 [-0.4%])</span> 2,307 |  2,579,903 | <span style='color: red'>(+2 [+0.4%])</span> 500 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/48d938a7433a6fd25390daebc632bebb8397c90b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/40375738890)
