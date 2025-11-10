#include <H5Cpp.h>
#include <vector>
#include <string>
#include <map>
#include <variant>
#include <type_traits>
#include <stdexcept>


namespace hdf5_utils
{

// Supported types
using H5DataVariant = std::variant<
    std::vector<int>,
    std::vector<float>,
    std::vector<double>,
    int,
    float,
    double,
    std::string
>;

void saveVariantToHDF5(H5::H5File& file, const std::string& key, const H5DataVariant& value)
{
    std::visit([&](auto&& data) {
        using T = std::decay_t<decltype(data)>;

        if constexpr (std::is_same_v<T, std::string>) {
            // Store string as a scalar dataset
            H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
            H5::DataSpace scalar_space(H5S_SCALAR);
            auto dataset = file.createDataSet(key, str_type, scalar_space);
            dataset.write(data, str_type);
        }
        else if constexpr (std::is_arithmetic_v<T>) {
            // Single numeric scalar
            H5::DataSpace scalar_space(H5S_SCALAR);
            H5::PredType type;
            if constexpr (std::is_same_v<T, int>) type = H5::PredType::NATIVE_INT;
            else if constexpr (std::is_same_v<T, float>) type = H5::PredType::NATIVE_FLOAT;
            else if constexpr (std::is_same_v<T, double>) type = H5::PredType::NATIVE_DOUBLE;
            else throw std::runtime_error("Unsupported scalar type");
            auto dataset = file.createDataSet(key, type, scalar_space);
            dataset.write(&data, type);
        }
        else if constexpr (std::is_same_v<T, std::vector<int>> ||
                           std::is_same_v<T, std::vector<float>> ||
                           std::is_same_v<T, std::vector<double>>) {
            // 1D numeric array
            const hsize_t dims[1] = { data.size() };
            H5::DataSpace space(1, dims);
            H5::PredType type;
            if constexpr (std::is_same_v<typename T::value_type, int>) type = H5::PredType::NATIVE_INT;
            else if constexpr (std::is_same_v<typename T::value_type, float>) type = H5::PredType::NATIVE_FLOAT;
            else if constexpr (std::is_same_v<typename T::value_type, double>) type = H5::PredType::NATIVE_DOUBLE;
            else throw std::runtime_error("Unsupported vector type");
            auto dataset = file.createDataSet(key, type, space);
            dataset.write(data.data(), type);
        }
        else {
            throw std::runtime_error("Unsupported data type in variant");
        }
    }, value);
}


void saveHeterogeneousData(const std::map<std::string, H5DataVariant>& items, const std::string& filepath) {
    H5::H5File file(filepath, H5F_ACC_TRUNC);
    for (const auto& [key, value] : items)
        saveVariantToHDF5(file, key, value);
}

} // namespace hdf5_utils