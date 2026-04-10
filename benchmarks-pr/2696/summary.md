| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2696/fibonacci-0a86e5607b8dcf941c531d03c402b2bddf60360c.md) | 3,831 |  12,000,265 |  959 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2696/keccak-0a86e5607b8dcf941c531d03c402b2bddf60360c.md) | 18,498 |  18,655,329 |  3,305 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2696/regex-0a86e5607b8dcf941c531d03c402b2bddf60360c.md) | 1,415 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2696/ecrecover-0a86e5607b8dcf941c531d03c402b2bddf60360c.md) | 645 |  123,583 |  278 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2696/pairing-0a86e5607b8dcf941c531d03c402b2bddf60360c.md) | 907 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2696/kitchen_sink-0a86e5607b8dcf941c531d03c402b2bddf60360c.md) | 2,147 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0a86e5607b8dcf941c531d03c402b2bddf60360c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24261182377)
