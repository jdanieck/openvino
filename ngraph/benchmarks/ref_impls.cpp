#include <numeric>
#include <vector>

#include <benchmark/benchmark.h>
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/reference/range.hpp"

static void range_ref_impl(benchmark::State& state)
{
    const int start = 0;
    const int step = 1;
    const size_t out_size = state.range(0);
    std::vector<int> out(out_size);

    for (auto _ : state)
    {
        ngraph::runtime::reference::range(&start, &step, out_size, out.data());
    }
}
BENCHMARK(range_ref_impl)->Unit(benchmark::kMicrosecond)->Range(1000, 10000000);

static void range_std_iota(benchmark::State& state)
{
    const int start = 0;
    const int step = 1;
    const size_t out_size = state.range(0);
    std::vector<int> out(state.range(0));

    for (auto _ : state)
    {
        std::iota(out.begin(), out.end(), start);
    }
}
BENCHMARK(range_std_iota)->Unit(benchmark::kMicrosecond)->Range(1000, 10000000);
