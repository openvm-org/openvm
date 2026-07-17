| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/fibonacci-c51e2bfe4ee8be94e51756f0f8f4b2b17d75cd91.md) |<span style='color: green'>(-68 [-4.2%])</span> 1,536 |  12,000,265 | <span style='color: red'>(+12 [+3.3%])</span> 376 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/keccak-c51e2bfe4ee8be94e51756f0f8f4b2b17d75cd91.md) |<span style='color: green'>(-1581 [-16.8%])</span> 7,823 |  18,655,329 | <span style='color: green'>(-5 [-0.3%])</span> 1,541 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/sha2_bench-c51e2bfe4ee8be94e51756f0f8f4b2b17d75cd91.md) |<span style='color: green'>(-308 [-6.3%])</span> 4,546 |  14,793,960 | <span style='color: red'>(+10 [+1.7%])</span> 584 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/regex-c51e2bfe4ee8be94e51756f0f8f4b2b17d75cd91.md) |<span style='color: red'>(+133 [+20.4%])</span> 786 |  4,137,067 | <span style='color: red'>(+5 [+2.3%])</span> 218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/ecrecover-c51e2bfe4ee8be94e51756f0f8f4b2b17d75cd91.md) |<span style='color: green'>(-15 [-3.4%])</span> 420 |  123,583 | <span style='color: red'>(+1 [+0.5%])</span> 186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/pairing-c51e2bfe4ee8be94e51756f0f8f4b2b17d75cd91.md) |<span style='color: green'>(-8 [-1.3%])</span> 590 |  1,745,757 | <span style='color: red'>(+2 [+1.0%])</span> 193 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/kitchen_sink-c51e2bfe4ee8be94e51756f0f8f4b2b17d75cd91.md) |<span style='color: red'>(+723 [+32.6%])</span> 2,940 |  2,579,903 | <span style='color: red'>(+6 [+1.3%])</span> 485 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c51e2bfe4ee8be94e51756f0f8f4b2b17d75cd91

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29552304235)
