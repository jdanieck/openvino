#include <numeric>
#include <vector>

#include <benchmark/benchmark.h>
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/reference/convolution.hpp"

using namespace ngraph;



static void convolution_1D_small_data(benchmark::State& state)
{
    const Strides strides{1};
    const CoordinateDiff padding{0};
    const Strides dilations{1};

    const Shape inputs_shape{1, 1, 6};
    std::vector<float> inputs(shape_size(inputs_shape));
    std::iota(inputs.begin(), inputs.end(), 0);

    const Shape filter_shape{1, 1, 3};
    const std::vector<float> filters{2.0f, 0.0f, 1.0f};

    const Shape outputs_shape{1, 1, 4};
    std::vector<float> out(shape_size(outputs_shape));
    
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(out.data());
        runtime::reference::convolution(inputs.data(),
                                        filters.data(),
                                        out.data(),
                                        inputs_shape,
                                        filter_shape,
                                        outputs_shape,
                                        strides,
                                        dilations,
                                        padding,
                                        padding,
                                        strides);

        benchmark::ClobberMemory();
    }
}
BENCHMARK(convolution_1D_small_data)->Unit(benchmark::kNanosecond);

static void convolution_1D_large_data(benchmark::State& state)
{
    const Strides strides{1};
    const CoordinateDiff padding{0};
    const Strides dilations{1};

    const Shape inputs_shape{10, 1, 100};
    std::vector<float> inputs(shape_size(inputs_shape));
    std::iota(inputs.begin(), inputs.end(), 0);

    const Shape filter_shape{1, 1, 3};
    const std::vector<float> filters{2.0f, 0.0f, 1.0f};

    const Shape outputs_shape{10, 1, 98};
    std::vector<float> out(shape_size(outputs_shape));

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(out.data());
        runtime::reference::convolution(inputs.data(),
                                        filters.data(),
                                        out.data(),
                                        inputs_shape,
                                        filter_shape,
                                        outputs_shape,
                                        strides,
                                        dilations,
                                        padding,
                                        padding,
                                        strides);

        benchmark::ClobberMemory();
    }
}
BENCHMARK(convolution_1D_large_data)->Unit(benchmark::kNanosecond);

static void convolution_2D_small_data(benchmark::State& state)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};

    const Shape inputs_shape{1, 1, 4, 4};
    std::vector<float> inputs(shape_size(inputs_shape));
    std::iota(inputs.begin(), inputs.end(), 0);

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filters{1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    3.0f, 2.0f, 1.0f};


    const Shape outputs_shape{1, 1, 2, 2};
    std::vector<float> out(shape_size(outputs_shape));

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(out.data());
        runtime::reference::convolution(inputs.data(),
                                        filters.data(),
                                        out.data(),
                                        inputs_shape,
                                        filter_shape,
                                        outputs_shape,
                                        strides,
                                        dilations,
                                        padding,
                                        padding,
                                        strides);

        benchmark::ClobberMemory();
    }
}
BENCHMARK(convolution_2D_small_data)->Unit(benchmark::kNanosecond);

static void convolution_2D_large_data(benchmark::State& state)
{
    const Strides strides{1, 1};
    const CoordinateDiff padding{0, 0};
    const Strides dilations{1, 1};

    const Shape inputs_shape{10, 1, 100, 100};
    std::vector<float> inputs(shape_size(inputs_shape));
    std::iota(inputs.begin(), inputs.end(), 0);

    const Shape filter_shape{1, 1, 3, 3};
    const std::vector<float> filters{1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    3.0f, 2.0f, 1.0f};

    const Shape outputs_shape{10, 1, 98, 98};
    std::vector<float> out(shape_size(outputs_shape));

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(out.data());
        runtime::reference::convolution(inputs.data(),
                                        filters.data(),
                                        out.data(),
                                        inputs_shape,
                                        filter_shape,
                                        outputs_shape,
                                        strides,
                                        dilations,
                                        padding,
                                        padding,
                                        strides);

        benchmark::ClobberMemory();
    }
}
BENCHMARK(convolution_2D_large_data)->Unit(benchmark::kNanosecond);

static void convolution_3D_small_data(benchmark::State& state)
{
    const Strides strides{1, 1, 1};
    const CoordinateDiff padding{0, 0, 0};
    const Strides dilations{1, 1, 1};

    const Shape inputs_shape{1, 1, 4, 4, 4};
    std::vector<float> inputs(shape_size(inputs_shape));
    std::iota(inputs.begin(), inputs.end(), 0);

    const Shape filter_shape{1, 1, 3, 3, 3};
     const std::vector<float> filters{                                    
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,                                    
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,                                    
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f};

    const Shape outputs_shape{1, 1, 2, 2, 2};
    std::vector<float> out(shape_size(outputs_shape));

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(out.data());
        runtime::reference::convolution(inputs.data(),
                                        filters.data(),
                                        out.data(),
                                        inputs_shape,
                                        filter_shape,
                                        outputs_shape,
                                        strides,
                                        dilations,
                                        padding,
                                        padding,
                                        strides);

        benchmark::ClobberMemory();
    }
}
BENCHMARK(convolution_3D_small_data)->Unit(benchmark::kNanosecond);

static void convolution_3D_large_data(benchmark::State& state)
{
    const Strides strides{1, 1, 1};
    const CoordinateDiff padding{0, 0, 0};
    const Strides dilations{1, 1, 1};

    const Shape inputs_shape{10, 1, 100, 100, 100};
    std::vector<float> inputs(shape_size(inputs_shape));
    std::iota(inputs.begin(), inputs.end(), 0);

    const Shape filter_shape{1, 1, 3, 3, 3};
     const std::vector<float> filters{                                    
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,                                    
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,                                    
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f};

    const Shape outputs_shape{10, 1, 98, 98, 98};
    std::vector<float> out(shape_size(outputs_shape));

    for (auto _ : state)
    {
        benchmark::DoNotOptimize(out.data());
        runtime::reference::convolution(inputs.data(),
                                        filters.data(),
                                        out.data(),
                                        inputs_shape,
                                        filter_shape,
                                        outputs_shape,
                                        strides,
                                        dilations,
                                        padding,
                                        padding,
                                        strides);

        benchmark::ClobberMemory();
    }
}
BENCHMARK(convolution_3D_large_data)->Unit(benchmark::kNanosecond);