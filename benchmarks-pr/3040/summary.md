| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/fibonacci-764bdc3400deb01b5811c531224e92b306a0ce22.md) | 418 |  4,000,051 |  239 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/keccak-764bdc3400deb01b5811c531224e92b306a0ce22.md) | 8,593 |  14,365,133 |  1,514 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/sha2_bench-764bdc3400deb01b5811c531224e92b306a0ce22.md) | 4,190 |  11,167,961 |  518 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/regex-764bdc3400deb01b5811c531224e92b306a0ce22.md) | 571 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/ecrecover-764bdc3400deb01b5811c531224e92b306a0ce22.md) | 221 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/pairing-764bdc3400deb01b5811c531224e92b306a0ce22.md) | 292 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/kitchen_sink-764bdc3400deb01b5811c531224e92b306a0ce22.md) | 1,935 |  1,979,971 |  467 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/764bdc3400deb01b5811c531224e92b306a0ce22

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29816410959)
