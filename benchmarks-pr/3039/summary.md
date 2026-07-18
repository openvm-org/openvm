| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/fibonacci-c6702bcdb231a7ad9ebe309b109c2ba9dac8c42d.md) | 420 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/keccak-c6702bcdb231a7ad9ebe309b109c2ba9dac8c42d.md) | 8,753 |  14,365,133 |  1,534 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/sha2_bench-c6702bcdb231a7ad9ebe309b109c2ba9dac8c42d.md) | 4,213 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/regex-c6702bcdb231a7ad9ebe309b109c2ba9dac8c42d.md) | 580 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/ecrecover-c6702bcdb231a7ad9ebe309b109c2ba9dac8c42d.md) | 217 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/pairing-c6702bcdb231a7ad9ebe309b109c2ba9dac8c42d.md) | 286 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/kitchen_sink-c6702bcdb231a7ad9ebe309b109c2ba9dac8c42d.md) | 1,934 |  1,979,971 |  467 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c6702bcdb231a7ad9ebe309b109c2ba9dac8c42d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29650556356)
