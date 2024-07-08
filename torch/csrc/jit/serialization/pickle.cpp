#include <torch/csrc/jit/serialization/pickle.h>

#include <ATen/core/ivalue.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/serialization/import_read.h>

namespace torch::jit {

void pickle(
    std::function<void(const char* data_start, size_t data_len)> writer,
    const IValue& ivalue,
    std::vector<at::Tensor>* tensor_table) {
  Pickler pickler(std::move(writer), tensor_table, nullptr, nullptr);
  pickler.protocol();
  pickler.pushIValue(ivalue);
  pickler.stop();
}

std::vector<char> pickle(
    const IValue& ivalue,
    std::vector<at::Tensor>* tensor_table) {
  std::vector<char> data;

  pickle(
      [&](const char* bytes, size_t len) {
        data.insert(data.end(), bytes, bytes + len);
      },
      ivalue,
      tensor_table);

  return data;
}

// This has to live here instead of the C++ API to mirror torch.save since the
// mobile build excludes the C++ API
std::vector<char> pickle_save(const at::IValue& ivalue) {
#ifndef C10_MOBILE
  // Pickle the IValue into an array of bytes
  std::vector<char> pickle_data;
  Pickler pickler([&](const char* buf, size_t size) {
    pickle_data.insert(pickle_data.end(), buf, buf + size);
  });
  pickler.protocol();
  pickler.pushIValue(ivalue);
  pickler.stop();

  std::vector<char> container_data;
  container_data.reserve(pickle_data.size());

  caffe2::serialize::PyTorchStreamWriter writer(
      [&](const void* void_bytes, size_t len) {
        const char* bytes = reinterpret_cast<const char*>(void_bytes);
        container_data.insert(container_data.end(), bytes, bytes + len);
        return len;
      });

  // Write the generated bytes and the associated tensors into a data.pkl file
  // and data/0, data/1, data/2... files for each of the tensors
  writeArchiveAndTensors(
      "data",
      pickle_data.data(),
      pickle_data.size(),
      pickler.tensorData(),
      writer);
  return container_data;
#else
  AT_ERROR(
      "pickle_save not supported on mobile "
      "(see https://github.com/pytorch/pytorch/pull/30108)");
#endif
}

#ifndef C10_MOBILE
size_t VectorReader::read(uint64_t pos, void* buf, size_t n, const char* what)
    const {
  std::copy(
      data_.data() + pos, data_.data() + pos + n, reinterpret_cast<char*>(buf));
  return n;
}
#endif

IValue pickle_load(const std::vector<char>& data) {
  // Read in the pickle data
#ifndef C10_MOBILE
  caffe2::serialize::PyTorchStreamReader reader(
      std::make_unique<VectorReader>(data));

  return readArchiveAndTensors(
      "data",
      /*pickle_prefix=*/"",
      /*tensor_prefix=*/"",
      /*type_resolver=*/std::nullopt,
      /*obj_loader=*/std::nullopt,
      /*device=*/std::nullopt,
      reader);
#else
  AT_ERROR(
      "pickle_load not supported on mobile "
      "(see https://github.com/pytorch/pytorch/pull/30108)");
#endif
};

IValue unpickle(
    std::function<size_t(char*, size_t)> reader,
    TypeResolver type_resolver,
    c10::ArrayRef<at::Tensor> tensor_table,
    c10::TypePtr (*type_parser)(const std::string&),
    ObjLoader obj_loader) {
  Unpickler unpickler(
      std::move(reader),
      std::move(type_resolver),
      tensor_table,
      std::move(obj_loader),
      type_parser);
  return unpickler.parse_ivalue();
}

IValue unpickle(
    const char* data,
    size_t size,
    TypeResolver type_resolver,
    c10::ArrayRef<at::Tensor> tensor_table,
    c10::TypePtr (*type_parser)(const std::string&)) {
  return unpickle(
      data, size, nullptr, std::move(type_resolver), tensor_table, type_parser);
}

IValue unpickle(
    const char* data,
    size_t size,
    ObjLoader obj_loader,
    TypeResolver type_resolver,
    c10::ArrayRef<at::Tensor> tensor_table,
    c10::TypePtr (*type_parser)(const std::string&)) {
  size_t bytes_read = 0;
  return unpickle(
      [&](char* buffer, size_t len) -> size_t {
        if (bytes_read >= size) {
          return 0;
        }
        len = std::min(size - bytes_read, len);
        // Copy len bytes into buffer
        const char* start = data + bytes_read;
        std::memcpy(buffer, start, len);
        bytes_read += len;
        return len;
      },
      std::move(type_resolver),
      tensor_table,
      type_parser,
      std::move(obj_loader));
}

} // namespace torch::jit
