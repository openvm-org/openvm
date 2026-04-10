| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2694/fibonacci-301a4fe3101b4be168f17f444b03893a6710602c.md) | 3,846 |  12,000,265 |  957 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2694/keccak-301a4fe3101b4be168f17f444b03893a6710602c.md) | 18,563 |  18,655,329 |  3,324 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2694/regex-301a4fe3101b4be168f17f444b03893a6710602c.md) | 1,400 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2694/ecrecover-301a4fe3101b4be168f17f444b03893a6710602c.md) | 647 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2694/pairing-301a4fe3101b4be168f17f444b03893a6710602c.md) | 910 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2694/kitchen_sink-301a4fe3101b4be168f17f444b03893a6710602c.md) | 2,159 |  2,579,903 |  438 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/301a4fe3101b4be168f17f444b03893a6710602c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24253022274)
