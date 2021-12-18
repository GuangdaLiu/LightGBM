#ifdef USE_CUDA

#include <LightGBM/dataset_loader.h>

namespace LightGBM {

__global__ void ValueToBinKernel(double* cuda_bin_upper_bounds_ptr[], int cuda_bin_upper_bounds_size[], bool cuda_should_feature_mapped[], double* cuda_batch_value_ptr[], data_size_t cur_cuda_batch_size) {
  data_size_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int col_inx = blockIdx.y;
  if (cuda_should_feature_mapped[col_inx]) {
    double value = cuda_batch_value_ptr[row_idx][col_inx];
    uint32_t bin = 0;
    // only consider NumericalBin now
    int l = 0;
    int r = cuda_bin_upper_bounds_size[col_inx];
    int m = 0;
    while (l < r) {
      m = (r + l - 1) / 2;
      if (value <= cuda_bin_upper_bounds_ptr[col_inx][m]) {
        r = m;
      } else {
        l = m + 1;
      }
    }
    bin = m;
  }
}

void DatasetLoader::LaunchValueToBinKernel(double* cuda_bin_upper_bounds_ptr[], int cuda_bin_upper_bounds_size[], bool cuda_should_feature_mapped[], double* cuda_batch_value_ptr[], data_size_t cur_cuda_batch_size, int num_total_features) {
  const int num_threads_per_block = 1024;
  int num_blocks_for_row = (cur_cuda_batch_size + num_threads_per_block - 1) / num_threads_per_block;
  dim3 num_blocks(num_blocks_for_row, num_total_features);
  ValueToBinKernel<<<num_blocks, num_threads_per_block>>>(cuda_bin_upper_bounds_ptr, cuda_bin_upper_bounds_size, cuda_should_feature_mapped, cuda_batch_value_ptr, cur_cuda_batch_size);
}
  
}

#endif