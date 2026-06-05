| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2838/fibonacci-6d52753f9a616d0b24dc19d847c770fb219c830a.md) | 3,766 |  12,000,265 |  920 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2838/keccak-6d52753f9a616d0b24dc19d847c770fb219c830a.md) | 17,650 |  18,655,329 |  3,204 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2838/sha2_bench-6d52753f9a616d0b24dc19d847c770fb219c830a.md) | 10,143 |  14,793,960 |  1,478 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2838/regex-6d52753f9a616d0b24dc19d847c770fb219c830a.md) | 1,400 |  4,137,067 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2838/ecrecover-6d52753f9a616d0b24dc19d847c770fb219c830a.md) | 603 |  123,583 |  249 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2838/pairing-6d52753f9a616d0b24dc19d847c770fb219c830a.md) | 879 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2838/kitchen_sink-6d52753f9a616d0b24dc19d847c770fb219c830a.md) | 3,865 |  2,579,903 |  951 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6d52753f9a616d0b24dc19d847c770fb219c830a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27029422055)
