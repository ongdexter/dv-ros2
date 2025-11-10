#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <random>

#include <NvInfer.h>

using namespace nvinfer1;

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

ICudaEngine* loadCudaEngine(const std::string& engineFilePath, IRuntime* runtime)
{
    // Read the serialized engine from file
    std::ifstream file(engineFilePath, std::ios::binary);
    if (!file)
    {
        std::cerr << "Error opening engine file: " << engineFilePath << std::endl;
        return nullptr;
    }

    file.seekg(0, file.end);
    size_t fsize = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engineData(fsize);
    file.read(engineData.data(), fsize);
    if (!file)
    {
        std::cerr << "Error reading engine file: " << engineFilePath << std::endl;
        return nullptr;
    }

    // Deserialize the engine
    ICudaEngine* engine_ptr = runtime->deserializeCudaEngine(engineData.data(), fsize);
    if (!engine_ptr)
    {
        std::cerr << "Error deserializing CUDA engine." << std::endl;
        return nullptr;
    }

    return engine_ptr;
}

IExecutionContext* createExecutionContext(ICudaEngine* engine)
{
    IExecutionContext* context = engine->createExecutionContext();
    if (!context)
    {
        std::cerr << "Error creating execution context." << std::endl;
        return nullptr;
    }

    /* TODO: is any of this necessary? */
    // context->setOptimizationProfile(0); // Set to the first profile if multiple profiles exist
    // context->setTensorAddress("input_voxelgrid", input_buffer);
    // context->setTensorAddress("output", output_buffer);
    // context->setInputShape("input_voxelgrid", Dims4{1, 18, 256, 320});
    // context->setOutputShape("output", Dims4{1, 2, 256, 320});

    return context;
}

void tensor_api(const ICudaEngine* engine)
{
    long unsigned int        nIO     = engine->getNbIOTensors();
    long unsigned int        nInput  = 0, nOutput = 0;
    std::vector<std::string> vTensorName(nIO);
    for (int i = 0; i < nIO; ++i)
    {
        vTensorName[i] = std::string(engine->getIOTensorName(i));
        nInput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kINPUT);
        nOutput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kOUTPUT);
    }
    std::cout << "Number of I/O tensors: " << nIO << " (Input: " << nInput << ", Output: " << nOutput << ")" << std::endl;
}

void inspect_buffers(const ICudaEngine* engine)
{
    int nBindings = engine->getNbBindings();
    for (int i = 0; i < nBindings; ++i) {
        std::string name = engine->getBindingName(i);
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        bool isInput = engine->bindingIsInput(i);

        std::cout << (isInput ? "Input" : "Output") << " binding " << i 
                << ": name=" << name << std::endl;
                // << ", dims=" << dims << std::endl;
    }
}

size_t getSizeByDim(const nvinfer1::Dims& dims) {
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i)
        size *= dims.d[i];
    return size;
}


void** allocate_buffers(const ICudaEngine* engine)
{
    int nBindings = engine->getNbBindings();
    void** buffers = new void*[nBindings];

    for (int i = 0; i < nBindings; ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        size_t size = getSizeByDim(dims);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);

        size_t eltSize;
        switch (dtype) {
            case nvinfer1::DataType::kFLOAT: eltSize = sizeof(float); break;
            // case nvinfer1::DataType::kHALF: eltSize = sizeof(__half); break;
            case nvinfer1::DataType::kINT8: eltSize = sizeof(int8_t); break;
            case nvinfer1::DataType::kINT32: eltSize = sizeof(int32_t); break;
            default: throw std::runtime_error("Unsupported datatype");
        }

        size_t bytes = size * eltSize;

        std::cout << "Allocating " << bytes << " bytes for binding " << i << std::endl;

        cudaMalloc(&buffers[i], bytes);
    }

    std::cout << "All buffers allocated." << std::endl;
    return buffers;
}


void move_data_to_buffer(
    void** buffers,
    const std::vector<float>& hostInput,
    size_t inputBytes
) {
    std::cout << "Copying input data to device." << std::endl;
    std::cout << "Input bytes: " << inputBytes << std::endl;
    std::cout << "Host input size: " << hostInput.size() << std::endl;
    std::cout << "Buffers root address: " << buffers << std::endl;
    std::cout << "Total buffers: " << sizeof(buffers) / sizeof(buffers[0]) << std::endl;
    std::cout << "Kind arg: " << cudaMemcpyHostToDevice << std::endl;


    auto buffer = buffers[0]; // Assuming input is at index 0
    cudaMemcpy(
        buffer,
        hostInput.data(),
        inputBytes,
        cudaMemcpyHostToDevice
    );
}

void fill_with_random(std::vector<float>& data, float min_val, float max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);

    for (auto& v : data) {
        v = dist(gen);
    }
}

std::vector<std::vector<float>> recover_model_outputs(const ICudaEngine* engine, void** buffers)
{
    std::vector<std::vector<float>> hostOutputs;
    std::cout << "Copying output back to host: " << engine->getNbBindings() << std::endl;
    for (int outputIndex = 0; outputIndex < engine->getNbBindings(); ++outputIndex)
    {
        if (engine->bindingIsInput(outputIndex))
        {
            std::cout << "Skipping input binding index: " << outputIndex << std::endl;
            continue;
        }

        std::cout << "Processing output binding index: " << outputIndex << std::endl;

        nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);
        size_t outputSize = getSizeByDim(outputDims);
        size_t outputBytes = outputSize * sizeof(float); // Assuming output is float
        std::vector<float> hostOutput(outputSize);
        
        cudaMemcpy(hostOutput.data(), buffers[outputIndex], outputBytes, cudaMemcpyDeviceToHost);

        std::cout << "Output start samples: ";
        for (int i = 0; i < 10; ++i)
            std::cout << hostOutput[i] << " ";
        std::cout << std::endl;

        std::cout << "Output end samples: ";
        for (int i = hostOutput.size() - 10; i < hostOutput.size(); ++i)
            std::cout << hostOutput[i] << " ";
        std::cout << std::endl;
        hostOutputs.push_back(std::move(hostOutput));
    }
    return hostOutputs;
}

void sanity_check(const std::vector<float>& hostInput, void** buffers, size_t inputBytes)
{
    /* This proved that moving data back and forth works fine, and allocation of random input values also worked. */
    std::vector<float> sanity_check(hostInput.size());
    cudaMemcpy(sanity_check.data(), buffers[0], inputBytes, cudaMemcpyDeviceToHost);
    std::cout << "Sanity check - first 10 and last 10 input samples on device:\n";
    for (int i = 0; i < 10; ++i)
        std::cout << sanity_check[i] << " | " << hostInput[i] << std::endl;
    for (int i = hostInput.size() - 10; i < hostInput.size(); ++i)
        std::cout << sanity_check[i] << " | " << hostInput[i] << std::endl;
}

void experiment(const std::string& engineFilePath)
{
    // Create the TensorRT runtime
    IRuntime* runtime = createInferRuntime(logger);
    if (!runtime)
    {
        std::cerr << "Error creating TensorRT runtime." << std::endl;
        return;
    }
    std::cout << "Successfully created TensorRT runtime." << std::endl;

    // Load the CUDA engine
    ICudaEngine* engine = loadCudaEngine(engineFilePath, runtime);
    if (!engine)
    {
        runtime->destroy();
        return;
    }
    std::cout << "Successfully loaded CUDA engine from: " << engineFilePath << std::endl;

    // Initialize everything
    tensor_api(engine);

    /* Optional: Check what's in the buffers */
    // inspect_buffers(engine);

    // Allocate device buffers
    auto buffers = allocate_buffers(engine);

    // Create execution context
    IExecutionContext *context = createExecutionContext(engine);
    if (!context)
    {
        engine->destroy();
        return;
    }
    std::cout << "Successfully created execution context." << std::endl;

    
    // Prepare input data
    nvinfer1::Dims input_dims = engine->getBindingDimensions(0); // Assuming input is at index 0
    size_t inputSize = getSizeByDim(input_dims);
    std::vector<float> hostInput(inputSize); // should be: 1 * 18 * 256 * 320 for evflownet
    fill_with_random(hostInput, 0.0f, 10.0f); // TODO: placeholder!

    size_t inputBytes = hostInput.size() * sizeof(hostInput[0]);
    std::cout << "Moving input bytes: " << inputBytes << " to device." << std::endl;
    move_data_to_buffer(buffers, hostInput, inputBytes);

    /* Optional: sanity check to verify data transfers to and from device (numbers should match) */
    // sanity_check(hostInput, buffers, inputBytes);

    // Execute the model - TODO: is the stream necessary?
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    context->executeV2(buffers);
    cudaStreamSynchronize(stream);

    // Retrieve outputs
    auto model_outputs = recover_model_outputs(engine, buffers);
    
    // Clean up
    std::cout << "Cleaning up." << std::endl;
    for (int i = 0; i < engine->getNbBindings(); ++i)
        cudaFree(buffers[i]);
    context->destroy();
    engine->destroy();
    runtime->destroy(); 
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <engine_file_path>" << std::endl;
        return 1;
    }

    std::string engineFilePath = argv[1];
    experiment(engineFilePath);

    return 0;
}
