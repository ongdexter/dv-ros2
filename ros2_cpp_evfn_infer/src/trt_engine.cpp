// trt_engine.cpp
#include "trt_engine.hpp"

TrtEngine::TrtEngine(std::string enginePath)
: engineFilePath_(std::move(enginePath))
{}

TrtEngine::~TrtEngine()
{
    // free device buffers
    for (void* p : deviceBuffers_) {
        if (p) cudaFree(p);
    }
    deviceBuffers_.clear();

    // API CHANGE
    // if (context_) { context_->destroy(); context_ = nullptr; }
    // if (engine_)  { engine_->destroy();  engine_ = nullptr; }
    // if (runtime_) { runtime_->destroy(); runtime_ = nullptr; }

    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void TrtEngine::load()
{
    // create runtime
    runtime_ = createInferRuntime(logger_);
    if (!runtime_) throw std::runtime_error("Failed to create TensorRT runtime.");

    // read engine file
    std::ifstream file(engineFilePath_, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Unable to open engine file: " + engineFilePath_);

    std::cout << "[TrtEngine::load] Loaded engine file: " << engineFilePath_ << std::endl;

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size <= 0) throw std::runtime_error("Engine file empty or invalid: " + engineFilePath_);

    std::vector<char> buffer(static_cast<size_t>(size));
    if (!file.read(buffer.data(), size)) throw std::runtime_error("Failed to read engine file: " + engineFilePath_);

    engine_ = runtime_->deserializeCudaEngine(buffer.data(), buffer.size());
    if (!engine_) throw std::runtime_error("Failed to deserialize CUDA engine.");

    context_ = engine_->createExecutionContext();
    if (!context_) throw std::runtime_error("Failed to create execution context.");

    nbBindings_ = engine_->getNbIOTensors();
}

Dims TrtEngine::getTensorDims(int bindingIndex) const
{
    if (!engine_) throw std::runtime_error("Engine not loaded.");
    if (bindingIndex < 0 || bindingIndex >= nbBindings_)
        throw std::out_of_range("bindingIndex out of range.");

    std::string name(engine_->getIOTensorName(bindingIndex));
    Dims dims = engine_->getTensorShape(name.c_str());
    return dims;
}

void TrtEngine::allocateBuffers()
{
    if (!engine_) throw std::runtime_error("Engine not loaded.");

    bindingByteSizes_.assign(nbBindings_, 0);
    deviceBuffers_.assign(nbBindings_, nullptr);

    for (int i = 0; i < nbBindings_; ++i) {
        Dims dims = getTensorDims(i);

        // caution: dynamic shapes may use -1 in dims. This code assumes static dims.
        if (dims.nbDims <= 0) {
            throw std::runtime_error("Dynamic binding shapes detected; this wrapper expects static shapes for binding " + std::to_string(i));
        }

        size_t eltCount = sizeByDims(dims);
        DataType dt = getBindingDataType(i);
        size_t bytes = eltCount * elementSize(dt);
        bindingByteSizes_[i] = bytes;

        cudaError_t cerr = cudaMalloc(&deviceBuffers_[i], bytes);
        if (cerr != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc failed for binding ") + std::to_string(i) + ": " + cudaGetErrorString(cerr));
        }
    }

    // create a stream for asynchronous operations (optional)
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
        throw std::runtime_error("cudaStreamCreate failed.");
    }
}

void TrtEngine::setInput(const std::vector<float>& hostInput, int bindingIndex)
{
    if (bindingIndex < 0 || bindingIndex >= nbBindings_)
        throw std::out_of_range("bindingIndex out of range.");

    if (!deviceBuffers_[bindingIndex]) throw std::runtime_error("Device buffer not allocated for binding " + std::to_string(bindingIndex));

    if (bindingByteSizes_[bindingIndex] != hostInput.size() * sizeof(float)) {
        // a mismatch — be explicit about assumptions:
        throw std::runtime_error("Host input size does not match binding byte size. host bytes="
            + std::to_string(hostInput.size()*sizeof(float)) + " expected=" + std::to_string(bindingByteSizes_[bindingIndex]));
    }

    cudaMemcpyAsync(deviceBuffers_[bindingIndex], hostInput.data(), bindingByteSizes_[bindingIndex], cudaMemcpyHostToDevice, stream_);
}

void TrtEngine::infer()
{
    if (!context_) throw std::runtime_error("Execution context not created.");
    // context->enqueueV2 or executeV2 can be used; executeV2 expects device pointers array
    // We use executeV2 which is simpler for static shapes:
    if (!deviceBuffers_.empty()) {
        void** rawBuffers = deviceBuffers_.data();
        // make sure to synchronize stream if using async copies
        if (!context_->executeV2(static_cast<void**>(rawBuffers))) {
            throw std::runtime_error("Execution failed (executeV2 returned false).");
        }
        // If you used async copies to stream_, you might want to synchronize here:
        cudaStreamSynchronize(stream_);
    } else {
        throw std::runtime_error("No device buffers allocated.");
    }
}

bool TrtEngine::isTensorInput(int bindingIndex) const
{
    if (!engine_) throw std::runtime_error("Engine not loaded.");
    if (bindingIndex < 0 || bindingIndex >= nbBindings_)
        throw std::out_of_range("bindingIndex out of range.");
    std::string name(engine_->getIOTensorName(bindingIndex));
    return engine_->getTensorIOMode(name.c_str()) == TensorIOMode::kINPUT;
}

bool TrtEngine::isTensorOutput(int bindingIndex) const
{
    if (!engine_) throw std::runtime_error("Engine not loaded.");
    if (bindingIndex < 0 || bindingIndex >= nbBindings_)
        throw std::out_of_range("bindingIndex out of range.");
    std::string name(engine_->getIOTensorName(bindingIndex));
    return engine_->getTensorIOMode(name.c_str()) == TensorIOMode::kOUTPUT;
}

DataType TrtEngine::getBindingDataType(int bindingIndex) const
{
    if (!engine_) throw std::runtime_error("Engine not loaded.");
    if (bindingIndex < 0 || bindingIndex >= nbBindings_)
        throw std::out_of_range("bindingIndex out of range.");
    std::string name(engine_->getIOTensorName(bindingIndex));
    return engine_->getTensorDataType(name.c_str());
}

std::vector<std::vector<float>> TrtEngine::getOutputs()
{
    std::vector<std::vector<float>> outputs;
    if (!engine_) throw std::runtime_error("Engine not loaded.");

    for (int i = 0; i < nbBindings_; ++i) {
        if (isTensorInput(i)) continue; // skip inputs

        size_t bytes = bindingByteSizes_[i];
        if (bytes == 0) throw std::runtime_error("Binding byte size is zero for binding " + std::to_string(i));

        // assume float outputs; check dtype and convert if needed
        auto dt = getBindingDataType(i);
        if (dt != DataType::kFLOAT) {
            throw std::runtime_error("Only float outputs are handled by getOutputs(); other dtypes not implemented.");
        }

        size_t nElements = bytes / sizeof(float);
        std::vector<float> hostOut(nElements);

        cudaMemcpyAsync(hostOut.data(), deviceBuffers_[i], bytes, cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        outputs.push_back(std::move(hostOut));
    }

    // each output comes pre-flattened, from 2 x H x W into a 1D vector
    return outputs;
}

void TrtEngine::loadSingleOutput(std::vector<float>& output, int bindingIndex)
{
    /* more efficient than getOutputs(), assuming correct pre-allocation */
    if (!engine_) throw std::runtime_error("Engine not loaded.");
    if (bindingIndex < 0 || bindingIndex >= nbBindings_)
        throw std::out_of_range("bindingIndex out of range.");

    if (isTensorInput(bindingIndex)) throw std::runtime_error("Binding index " + std::to_string(bindingIndex) + " is an input, expected output.");

    if (!deviceBuffers_[bindingIndex]) throw std::runtime_error("Device buffer not allocated for binding " + std::to_string(bindingIndex));

    if (!stream_) throw std::runtime_error("CUDA stream not created.");

    if (bindingByteSizes_[bindingIndex] != output.size() * sizeof(float)) {
        // a mismatch — be explicit about assumptions:
        throw std::runtime_error("Host output size does not match binding byte size. host bytes="
            + std::to_string(output.size()*sizeof(float)) + " expected=" + std::to_string(bindingByteSizes_[bindingIndex]));
    }

    cudaMemcpyAsync(output.data(), deviceBuffers_[bindingIndex], bindingByteSizes_[bindingIndex], cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
}

void TrtEngine::printIOInfo() const
{
    if (!engine_) {
        std::cout << "Engine not loaded." << std::endl;
        return;
    }
    auto nIO = engine_->getNbIOTensors();
    auto nInput = 0, nOutput = 0;
    for (auto i = 0; i < nIO; ++i) {
        std::string name(engine_->getIOTensorName(i));
        if (engine_->getTensorIOMode(name.c_str()) == TensorIOMode::kINPUT) ++nInput;
        else ++nOutput;
    }
    std::cout << "I/O tensors: " << nIO << " (Inputs: " << nInput << ", Outputs: " << nOutput << ")" << std::endl;
}

void TrtEngine::printBindings() const
{
    if (!engine_) { std::cout << "Engine not loaded." << std::endl; return; }
    for (auto i = 0; i < nbBindings_; ++i) {
        std::string name(engine_->getIOTensorName(i));
        std::cout << (isTensorInput(i) ? "Input " : "Output ")
                  << "binding " << i << " name=" << name << std::endl;
    }
}

size_t TrtEngine::sizeByDims(const Dims& d) const noexcept
{
    size_t s = 1;
    for (auto i = 0; i < d.nbDims; ++i) s *= (d.d[i] <= 0 ? 1 : d.d[i]); // treat unknown as 1 for fallback (but we raise earlier)
    return s;
}

size_t TrtEngine::elementSize(DataType dt) const {
    switch (dt) {
        case DataType::kFLOAT: return sizeof(float);
        case DataType::kHALF:  return sizeof(uint16_t); // __half storage
        case DataType::kINT8:  return sizeof(int8_t);
        case DataType::kINT32: return sizeof(int32_t);
        default: throw std::runtime_error("Unsupported DataType");
    }
}
