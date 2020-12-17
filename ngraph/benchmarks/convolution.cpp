#include <numeric>
#include <vector>

#include <benchmark/benchmark.h>
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/reference/convolution.hpp"

using namespace ngraph;

static void convolution_ref_impl(benchmark::State& state)
{
    size_t data_size = state.range(0);
    std::vector<float> in(data_size);
    std::vector<float> out(data_size);
    std::iota(begin(in), end(in), 1);

    const std::vector<float> filter{3.0f, 3.0f, 3.0f, 3.0f};
    const ngraph::Shape in_shape{1, 2, 2, 2};
    const ngraph::Shape filter_shape{2, 2, 1, 1};
    const ngraph::Shape out_shape{1, 2, 2, 2};
    const Strides& stride = Strides{1, 1};
    const Strides& filter_dilation = Strides{1, 1};
    const CoordinateDiff& in_pad_below = CoordinateDiff{0, 0};
    const CoordinateDiff& in_pad_above = CoordinateDiff{0, 0};
    const Strides& in_dilation = Strides{1, 1};

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(out.data());
        runtime::reference::convolution(in.data(),
                                        filter.data(),
                                        out.data(),
                                        in_shape,
                                        filter_shape,
                                        out_shape,
                                        stride,
                                        filter_dilation,
                                        in_pad_below,
                                        in_pad_above,
                                        in_dilation);

        benchmark::ClobberMemory();
    }
}
BENCHMARK(convolution_ref_impl)->Unit(benchmark::kNanosecond)->Range(8, 8 << 24);
