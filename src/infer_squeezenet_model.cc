#include <array>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>

#include "absl/cleanup/cleanup.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_experimental.h"

static TF_Buffer* ReadBufferFromFile(const char* file_path) {
  std::ifstream buffer_file(file_path, std::ifstream::binary);
  if (buffer_file.fail()) {
    return nullptr;
  }

  if (!buffer_file.is_open()) {
    return nullptr;
  }

  if (buffer_file.seekg(0, std::ifstream::end).fail()) {
    return nullptr;
  }

  int num_bytes = buffer_file.tellg();
  if (num_bytes <= 0) {
    return nullptr;
  }

  if (buffer_file.seekg(0, std::ifstream::beg).fail()) {
    return nullptr;
  }

  char* bytes = new char[num_bytes];
  if (buffer_file.read(bytes, num_bytes).fail()) {
    return nullptr;
  }

  auto buffer = TF_NewBuffer();
  buffer->data = bytes;
  buffer->length = num_bytes;
  buffer->data_deallocator = [](void* bytes, size_t num_bytes) {
    delete[] static_cast<char*>(bytes);
  };

  return buffer;
}

int main() {
  TF_Status* status = TF_NewStatus();
  auto status_cleanup =
      absl::MakeCleanup([status] { TF_DeleteStatus(status); });

  // Read the frozen graph from the file
  TF_Buffer* buffer = ReadBufferFromFile("squeezenet_model/squeezenet.pb");

  if (buffer == nullptr) {
    throw std::invalid_argument("Error: buffer is null");
  }

  auto buffer_cleanup =
      absl::MakeCleanup([buffer] { TF_DeleteBuffer(buffer); });

  auto graph = TF_NewGraph();
  auto graph_cleanup = absl::MakeCleanup([graph] { TF_DeleteGraph(graph); });

  auto options = TF_NewImportGraphDefOptions();
  auto options_cleanup =
      absl::MakeCleanup([options] { TF_DeleteImportGraphDefOptions(options); });

  TF_ImportGraphDefOptionsSetDefaultDevice(options, "/device:DML:0");
  TF_GraphImportGraphDef(graph, buffer, options, status);

  if (TF_GetCode(status) != TF_OK) {
    throw std::invalid_argument("Error: " + std::string(TF_Message(status)));
  }

  auto input_op = TF_Output{TF_GraphOperationByName(graph, "input_1"), 0};

  if (input_op.oper == nullptr) {
    throw std::invalid_argument("Error: input_op.oper is null");
  }

  int num_input_dims = TF_GraphGetTensorNumDims(graph, input_op, status);

  if (TF_GetCode(status) != TF_OK) {
    throw std::invalid_argument("Error: " + std::string(TF_Message(status)));
  }

  std::vector<int64_t> input_dims(num_input_dims);
  TF_GraphGetTensorShape(graph, input_op, input_dims.data(), num_input_dims,
                         status);

  if (TF_GetCode(status) != TF_OK) {
    throw std::invalid_argument("Error: " + std::string(TF_Message(status)));
  }

  // Set a batch dimension of 1 if the first dimension is free
  if (input_dims[0] == -1) {
    input_dims[0] = 1;
  }

  int64_t num_input_elements = std::accumulate(
      input_dims.begin(), input_dims.end(), 1LLU, std::multiplies<int64_t>());

  TF_DataType input_dtype = TF_OperationOutputType(input_op);

  // Generate random input values with a fixed seed
  std::random_device random_device;
  std::mt19937 generator(random_device());
  generator.seed(42);
  std::uniform_real_distribution<> distribution(0.0f, 1.0f);

  float* input_vals = new float[num_input_elements];
  for (int i = 0; i < num_input_elements; ++i) {
    input_vals[i] = distribution(generator);
  }

  // Create a tensor with the generated values
  auto input_tensor = TF_NewTensor(
      TF_FLOAT, input_dims.data(), num_input_dims, input_vals,
      num_input_elements * sizeof(float),
      [](void* data, size_t len, void* arg) {
        delete[] static_cast<float*>(data);
      },
      nullptr);
  if (input_tensor == nullptr) {
    throw std::invalid_argument("Error: input_tensor is null");
  }

  auto input_tensor_cleanup =
      absl::MakeCleanup([input_tensor] { TF_DeleteTensor(input_tensor); });

  TF_Tensor* output_tensor = nullptr;
  auto output_tensor_cleanup =
      absl::MakeCleanup([output_tensor] { TF_DeleteTensor(output_tensor); });

  auto output_op = TF_Output{TF_GraphOperationByName(graph, "loss/Softmax"), 0};

  // Build the session
  auto session_options = TF_NewSessionOptions();
  auto session_options_cleanup = absl::MakeCleanup(
      [session_options] { TF_DeleteSessionOptions(session_options); });

  auto session = TF_NewSession(graph, session_options, status);
  if (TF_GetCode(status) != TF_OK) {
    throw std::invalid_argument("Error: " + std::string(TF_Message(status)));
  }

  // Run the session
  TF_SessionRun(session, nullptr, &input_op, &input_tensor, 1, &output_op,
                &output_tensor, 1, nullptr, 0, nullptr, status);
  if (TF_GetCode(status) != TF_OK) {
    throw std::invalid_argument("Error: " + std::string(TF_Message(status)));
  }

  TF_CloseSession(session, status);
  if (TF_GetCode(status) != TF_OK) {
    throw std::invalid_argument("Error: " + std::string(TF_Message(status)));
  }

  TF_DeleteSession(session, status);
  if (TF_GetCode(status) != TF_OK) {
    throw std::invalid_argument("Error: " + std::string(TF_Message(status)));
  }

  if (output_tensor == nullptr) {
    throw std::invalid_argument("Error: output_tensor is null");
  }

  int64_t num_output_elements = TF_TensorElementCount(output_tensor);
  void* raw_output_data = TF_TensorData(output_tensor);
  float* float_output_data = static_cast<float*>(raw_output_data);

  printf("Output tensor: \n");
  for (int64_t i = 0; i < num_output_elements; ++i) {
    printf("%f, ", float_output_data[i]);
  }
  printf("\n");

  return 0;
}
