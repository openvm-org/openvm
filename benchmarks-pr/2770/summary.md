| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/fibonacci-f9385b103bfddcf54d345ee3c0a868940ccf7ccd.md) | 3,809 |  12,000,265 |  948 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/keccak-f9385b103bfddcf54d345ee3c0a868940ccf7ccd.md) | 18,694 |  18,655,329 |  3,335 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/sha2_bench-f9385b103bfddcf54d345ee3c0a868940ccf7ccd.md) | 9,204 |  14,793,960 |  1,415 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/regex-f9385b103bfddcf54d345ee3c0a868940ccf7ccd.md) | 1,399 |  4,137,067 |  372 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/ecrecover-f9385b103bfddcf54d345ee3c0a868940ccf7ccd.md) | 641 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/pairing-f9385b103bfddcf54d345ee3c0a868940ccf7ccd.md) | 895 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/kitchen_sink-f9385b103bfddcf54d345ee3c0a868940ccf7ccd.md) | 2,089 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f9385b103bfddcf54d345ee3c0a868940ccf7ccd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25540073754)
