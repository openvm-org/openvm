| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/fibonacci-e735be993f4cf814e42b91e159da41cd48b6f472.md) |<span style='color: green'>(-5 [-0.3%])</span> 1,690 |  12,000,265 |  370 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/keccak-e735be993f4cf814e42b91e159da41cd48b6f472.md) |<span style='color: green'>(-100 [-1.0%])</span> 9,647 |  18,655,329 | <span style='color: red'>(+8 [+0.5%])</span> 1,552 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/sha2_bench-e735be993f4cf814e42b91e159da41cd48b6f472.md) |<span style='color: green'>(-37 [-0.7%])</span> 5,207 |  14,793,960 | <span style='color: green'>(-3 [-0.5%])</span> 582 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/regex-e735be993f4cf814e42b91e159da41cd48b6f472.md) |<span style='color: red'>(+4 [+0.6%])</span> 693 |  4,137,067 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/ecrecover-e735be993f4cf814e42b91e159da41cd48b6f472.md) |<span style='color: red'>(+4 [+0.9%])</span> 438 |  123,583 | <span style='color: red'>(+1 [+0.5%])</span> 191 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/pairing-e735be993f4cf814e42b91e159da41cd48b6f472.md) |<span style='color: green'>(-46 [-7.4%])</span> 578 |  1,745,757 | <span style='color: green'>(-1 [-0.5%])</span> 194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3160/kitchen_sink-e735be993f4cf814e42b91e159da41cd48b6f472.md) |<span style='color: green'>(-20 [-0.9%])</span> 2,286 |  2,579,903 | <span style='color: green'>(-7 [-1.4%])</span> 492 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e735be993f4cf814e42b91e159da41cd48b6f472

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/40330733671)
