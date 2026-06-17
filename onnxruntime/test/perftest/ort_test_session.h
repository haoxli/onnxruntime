// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <core/session/onnxruntime_cxx_api.h>
#include <random>
#include "test_configuration.h"
#include "test_session.h"

#if defined(USE_CUDA) || defined(USE_TENSORRT) || defined(USE_NV)
#include <cuda_runtime.h>
#endif

class TestModelInfo;
namespace onnxruntime {
namespace perftest {
class OnnxRuntimeTestSession : public TestSession {
 public:
  OnnxRuntimeTestSession(Ort::Env& env, std::random_device& rd, const PerformanceTestConfig& performance_test_config,
                         const TestModelInfo& m);

#ifdef USE_WEBGPU
  // Lazily set up the persistent IO binding with GPU input/output tensors and CPU staging buffers
  // used for WebGPU graph capture/replay. Called on the first inference once the CPU inputs are
  // available (inputs are preloaded after construction).
  void InitializeWebGpuIoBinding();
  // Run inference using the WebGPU IO binding and copy GPU outputs back to CPU to simulate real usage.
  RunTiming RunWithWebGpuIoBinding();
#endif

  void PreLoadTestData(size_t test_data_id, size_t input_id, Ort::Value&& value) override {
    if (test_inputs_.size() < test_data_id + 1) {
      test_inputs_.resize(test_data_id + 1);
    }
    if (test_inputs_.at(test_data_id).size() == 0) {
      for (int i = 0; i < input_length_; i++)
        test_inputs_[test_data_id].emplace_back(nullptr);
    }
    test_inputs_[test_data_id][input_id] = std::move(value);
  }

  bool PopulateGeneratedInputTestData(int32_t seed);

  ~OnnxRuntimeTestSession();

  RunTiming Run() override;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OnnxRuntimeTestSession);

 private:
  Ort::Session session_{nullptr};
  std::mt19937 rand_engine_;
  std::uniform_int_distribution<int> dist_;
  Ort::AllocatorWithDefaultOptions default_allocator_;
  // Note: custom_allocator_, if used, must outlive the `Ort::Value`s allocated with it in test_inputs_ and outputs_.
  // and must be declared before them to ensure it is destroyed after them.
  Ort::Allocator custom_allocator_{nullptr};
  Ort::UnownedAllocator allocator_{default_allocator_};
  std::vector<std::vector<Ort::Value>> test_inputs_;
  std::vector<Ort::Value> outputs_;
  std::vector<std::string> output_names_;
  // The same size with output_names_.
  // TODO: implement a customized allocator, then we can remove output_names_ to simplify this code
  std::vector<const char*> output_names_raw_ptr;
  std::vector<const char*> input_names_;
  std::vector<std::string> input_names_str_;
  const int input_length_;
  std::string provider_name_;
  std::string device_memory_name_;  // Device memory type name to use from the list in allocator.h
  const std::unordered_map<std::string, std::string>& run_config_entries_;
  bool has_dynamic_output_shapes_ = false;
#ifdef USE_WEBGPU
  // WebGPU graph capture support. When enabled, inference uses a persistent IO binding with all
  // inputs/outputs in preallocated GPU memory (required so captured commands reference stable
  // buffers across replays). The GPU outputs are copied back to CPU each run to simulate real usage.
  Ort::Env* env_ = nullptr;  // Used for cross-device (GPU<->CPU) tensor copies via OrtApi::CopyTensors.
  bool enable_webgpu_graph_capture_ = false;
  Ort::IoBinding webgpu_io_binding_{nullptr};
  std::vector<Ort::Value> webgpu_gpu_inputs_;   // persistent GPU input buffers bound once
  std::vector<Ort::Value> webgpu_cpu_outputs_;  // CPU staging tensors that receive the GPU outputs
#endif
#if defined(USE_CUDA) || defined(USE_TENSORRT) || defined(USE_NV)
  cudaStream_t stream_;  // Device stream if required by IO bindings
#endif
  Ort::ArenaCfg cuda_mempool_arena_cfg_{nullptr};
};

}  // namespace perftest
}  // namespace onnxruntime
