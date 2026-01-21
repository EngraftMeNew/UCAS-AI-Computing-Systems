//Pytorch扩展头文件的引用
#include <torch/extension.h>
#include <cmath>
#include <cstring>
using namespace std;

//mysigmoid_cpu函数的具体实现
torch::Tensor mysigmoid_cpu(const torch::Tensor & dets) {
  //TODO: 将输入的tensor转化为浮点类型的vector
  auto input = dets.contiguous().to(torch::kCPU).to(torch::kFloat32);
  const float* input_ptr = input.data_ptr<float>();
  std::vector<float> input_data(input.numel());
  std::memcpy(input_data.data(), input_ptr, input.numel() * sizeof(float));
  int input_size = input_data.size(); 
  //TODO: 创建一个浮点类型的output_data，output_data为大小与输入相同的vector
  std::vector<float> output_data(input_size);
  //TODO: 对于输入向量的每个元素计算mysigmoid
  for (int i = 0; i < input_size; ++i) {
    const float x = input_data[i];
    // Standard sigmoid
    output_data[i] = 1.0f / (1.0f + std::exp(-x));
  }
  //TODO: Create tensor options with dtype float32
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  //TODO: Create a tensor from the output vector
  auto foo= torch::from_blob(output_data.data(), {int64_t(output_data.size())}, opts).clone();
  //TODO: 将得到的tensor转换成所需的大小
  auto output = foo.view(input.sizes());
  return output;
} 
//TODO: 算子绑定为Pytorch的模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {	// 绑定部分
  m.def("mysigmoid", &mysigmoid_cpu, "MySigmoid activation (CPU)");
}       
