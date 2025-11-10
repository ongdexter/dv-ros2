// main.cpp
#include "trt_engine.hpp"

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <engine_file>\n";
        return 1;
    }

    try {
        TrtEngine engine(argv[1]);
        engine.load();
        engine.printIOInfo();
        engine.printBindings();

        engine.allocateBuffers();

        // prepare a dummy input based on binding 0 dims (this assumes binding 0 is input & float)
        // For a real app you must read the dims and generate/prepare matching data.
        // Here we attempt to get binding 0 dims via the engine - if different, adjust accordingly.
        // NOTE: This example assumes binding 0 byte size matches floats and is an input.
        // Retrieve the size via engine internals is possible but kept simple here.

        // For demonstration we assume input binding is index 0 and is float:
        // compute number of floats from internal buffer size.
        const size_t dummyElems = engine.getBindingSizeBytes(0) / sizeof(float);
        
        /* random input */
        /*
        std::vector<float> hostInput(dummyElems);
        // fill with simple pattern
        std::mt19937 gen(123);
        std::uniform_real_distribution<float> d(0.0f, 1.0f);
        for (auto &x : hostInput) x = d(gen);
        */

        /* static test input */
        Eigen::Tensor<float, 4, Eigen::RowMajor> testInputTensor(1, 18, 256, 320); // example shape
        testInputTensor.setZero();
        
        for (auto t = 0; t < testInputTensor.dimension(1); t += 2)
            for (auto h = 0; h < testInputTensor.dimension(2); h += 10)
                for (auto w = 0; w < testInputTensor.dimension(3); w += 10)
                    testInputTensor(0, t, h, w) = 1.0;
        
        std::vector<float> hostInput(dummyElems);
        std::copy_n(testInputTensor.data(), dummyElems, hostInput.data());

        engine.setInput(hostInput, 0);
        engine.infer();
        auto outputs = engine.getOutputs();

        std::cout << "Number of outputs: " << outputs.size() << '\n';
        for (size_t i = 0; i < outputs.size(); ++i) {
            std::cout << "Output[" << i << "] size=" << outputs[i].size() << " first samples: ";
            for (size_t s = 0; s < std::min<size_t>(10, outputs[i].size()); ++s) std::cout << outputs[i][s] << " ";
            std::cout << std::endl;
        }

        Eigen::DSizes<long, 3> dimensions(2, 256, 320);
        Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>> my_tensor(outputs[2].data(), dimensions);

        std::cout << "Output[2] tensor shape: "
                  << my_tensor.dimension(0) << " x "
                  << my_tensor.dimension(1) << " x "
                  << my_tensor.dimension(2) << std::endl;

        std::cout << "Output tensor sample values at (0,0,:10): ";
        for (int w = 10; w >= 1; --w) {
            std::cout << my_tensor(1, 255, 320-w) << " ";
        }
        std::cout << std::endl;




        std::cout << "Checking tensor memory manangement" << std::endl;

        std::vector<int> temp;
        for (int i = 0; i < 100; ++i) {
            temp.push_back(i);
        }
        Eigen::DSizes<long, 3> test_dims(2, 5, 10);
        Eigen::TensorMap<Eigen::Tensor<int, 3, Eigen::RowMajor>> tensor_map(temp.data(), test_dims);

        std::cout << "Tensor map shape: "
                  << tensor_map.dimension(0) << " x "
                  << tensor_map.dimension(1) << " x "
                  << tensor_map.dimension(2) << std::endl;
        std::cout << "Tensor map sample values at (0,0,:10): ";
        for (int w = 0; w < 10; ++w) {
            std::cout << tensor_map(0, 0, w) << " ";
        }
        std::cout << std::endl;

    }
    catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return 2;
    }

    return 0;
}
