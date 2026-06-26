| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2940/fibonacci-943604ada64a0199971f9816d2095ca9127cd810.md) | 3,082 |  12,000,265 |  679 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2940/keccak-943604ada64a0199971f9816d2095ca9127cd810.md) | 16,274 |  18,655,329 |  3,010 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2940/sha2_bench-943604ada64a0199971f9816d2095ca9127cd810.md) | 9,118 |  14,793,960 |  1,108 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2940/regex-943604ada64a0199971f9816d2095ca9127cd810.md) | 1,177 |  4,137,067 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2940/ecrecover-943604ada64a0199971f9816d2095ca9127cd810.md) | 602 |  123,583 |  292 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2940/pairing-943604ada64a0199971f9816d2095ca9127cd810.md) | 935 |  1,745,757 |  304 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2940/kitchen_sink-943604ada64a0199971f9816d2095ca9127cd810.md) | 4,137 |  2,579,903 |  889 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/943604ada64a0199971f9816d2095ca9127cd810

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28271097274)
