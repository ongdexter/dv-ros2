// trt_engine.hpp
#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <random>
#include <memory>

using namespace nvinfer1;

// Simple logger (only warnings and errors)
class Logger : public ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << std::endl;
    }
};

// RAII wrapper for TensorRT inference
class TrtEngine
{
public:
    // Construct with the serialized engine path
    explicit TrtEngine(std::string enginePath);
    ~TrtEngine();

    // non-copyable
    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    // Allow move
    TrtEngine(TrtEngine&&) noexcept = default;
    TrtEngine& operator=(TrtEngine&&) noexcept = default;

    // Load & deserialize the engine (throws on error)
    void load();

    // Allocate device buffers for all bindings (throws on error)
    void allocateBuffers();

    size_t getBindingSizeBytes(int bindingIndex) const
    {
        if (bindingIndex < 0 || bindingIndex >= nbBindings_)
            throw std::out_of_range("bindingIndex out of range.");
        return bindingByteSizes_[bindingIndex];
    }

    // Fill the device input buffer from a host vector (assumes first input binding index)
    // If your model has multiple input bindings extend accordingly.
    void setInput(const std::vector<float>& hostInput, int bindingIndex = 0);

    // Run synchronous inference (throws on error)
    void infer();

    // Copy outputs back to host and return them as a vector per output-binding
    std::vector<std::vector<float>> getOutputs();

    // efficient single output loader into preallocated vector
    void loadSingleOutput(std::vector<float>& output, int bindingIndex);

    // Utility inspection helpers
    void printIOInfo() const;
    void printBindings() const;

private:
    // helper functions
    size_t sizeByDims(const Dims& d) const noexcept;
    size_t elementSize(DataType dt) const;
    int firstInputBinding() const noexcept;
    std::vector<int> outputBindingIndices() const noexcept;

    // API compatibility wrappers
    bool isTensorInput(int bindingIndex) const;
    bool isTensorOutput(int bindingIndex) const;
    DataType getBindingDataType(int bindingIndex) const;
    Dims getTensorDims(int bindingIndex) const;

    // members
    Logger logger_;
    std::string engineFilePath_;

    IRuntime* runtime_{nullptr};
    ICudaEngine* engine_{nullptr};
    IExecutionContext* context_{nullptr};

    std::vector<void*> deviceBuffers_;        // device pointers (size == nbBindings)
    std::vector<size_t> bindingByteSizes_;    // same length as nbBindings
    int nbBindings_{0};

    cudaStream_t stream_{nullptr};
};
