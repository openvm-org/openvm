| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3142/fibonacci-45b9c093943fb63e4941f1012bd99037e14f82d7.md) |<span style='color: green'>(-8 [-0.5%])</span> 1,669 |  12,000,265 | <span style='color: red'>(+1 [+0.3%])</span> 371 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3142/keccak-45b9c093943fb63e4941f1012bd99037e14f82d7.md) |<span style='color: red'>(+121 [+1.3%])</span> 9,656 |  18,655,329 | <span style='color: red'>(+5 [+0.3%])</span> 1,548 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3142/sha2_bench-45b9c093943fb63e4941f1012bd99037e14f82d7.md) |<span style='color: red'>(+34 [+0.6%])</span> 5,334 |  14,793,960 | <span style='color: red'>(+4 [+0.7%])</span> 596 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3142/regex-45b9c093943fb63e4941f1012bd99037e14f82d7.md) |<span style='color: red'>(+1 [+0.1%])</span> 704 |  4,137,067 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3142/ecrecover-45b9c093943fb63e4941f1012bd99037e14f82d7.md) |<span style='color: red'>(+8 [+1.9%])</span> 438 |  123,583 | <span style='color: red'>(+2 [+1.1%])</span> 191 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3142/pairing-45b9c093943fb63e4941f1012bd99037e14f82d7.md) |<span style='color: red'>(+33 [+5.6%])</span> 619 |  1,745,757 | <span style='color: red'>(+2 [+1.0%])</span> 196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3142/kitchen_sink-45b9c093943fb63e4941f1012bd99037e14f82d7.md) |<span style='color: red'>(+23 [+1.0%])</span> 2,321 |  2,579,903 | <span style='color: red'>(+9 [+1.8%])</span> 502 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/45b9c093943fb63e4941f1012bd99037e14f82d7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33407086452)
