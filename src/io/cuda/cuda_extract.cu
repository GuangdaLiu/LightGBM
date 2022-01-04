#ifdef USE_CUDA

#include <LightGBM/dataset_loader.h>

namespace LightGBM {

__global__ void ValueToBinKernel(uint32_t* cuda_batch_bins_ptr[], double* cuda_bin_upper_bounds_ptr[], const int cuda_bin_upper_bounds_size[], 
                                const bool cuda_should_feature_mapped[], double* cuda_batch_value_ptr[], const data_size_t cur_cuda_batch_size,
                                const int cuda_feature2group[], const int cuda_feature2subfeature[], const bool cuda_groups_is_multi_val[], const uint32_t cuda_most_freq_bins[],
                                const bool cuda_bin_type_is_numerical[], const bool cuda_missing_type_is_nan[], unsigned int* cuda_categorical_2_bin_ptr[]) {
  data_size_t row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int col_idx = blockIdx.y;
  if (row_idx < cur_cuda_batch_size && cuda_should_feature_mapped[col_idx]) {
    double value = cuda_batch_value_ptr[row_idx][col_idx];
    uint32_t bin = 0;
    if (cuda_bin_type_is_numerical[col_idx]) {
      int l = 0;
      int r = cuda_bin_upper_bounds_size[col_idx];
      int m = 0;
      while (l < r) {
        m = (r + l - 1) / 2;
        if (value <= cuda_bin_upper_bounds_ptr[col_idx][m]) {
          r = m;
        } else {
          l = m + 1;
        }
      }
      bin = m;
    } else {
      int int_value = static_cast<int>(value);
      if (int_value < 0) {
        bin = 0;
      }
      bin = cuda_categorical_2_bin_ptr[col_idx][int_value];
    }
    uint32_t most_freq_bin = cuda_most_freq_bins[col_idx];
    if (bin == most_freq_bin) {
      return;
    }
    if (most_freq_bin == 0) {
      bin -= 1;
    }
    cuda_batch_bins_ptr[row_idx][col_idx] = bin;
  }
}

void DatasetLoader::LaunchValueToBinKernel(uint32_t* cuda_batch_bins_ptr[], double* cuda_bin_upper_bounds_ptr[], const int cuda_bin_upper_bounds_size[], 
                                          const bool cuda_should_feature_mapped[], double* cuda_batch_value_ptr[], const data_size_t cur_cuda_batch_size, const int num_features,
                                          const int cuda_feature2group[], const int cuda_feature2subfeature[], const bool cuda_groups_is_multi_val[], const uint32_t cuda_most_freq_bins[],
                                          const bool cuda_bin_type_is_numerical[], const bool cuda_missing_type_is_nan[], unsigned int* cuda_categorical_2_bin_ptr[]) {
  const int num_threads_per_block = 1024;
  int num_blocks_for_row = (cur_cuda_batch_size + num_threads_per_block - 1) / num_threads_per_block;
  dim3 num_blocks(num_blocks_for_row, num_features);
  ValueToBinKernel<<<num_blocks, num_threads_per_block>>>(cuda_batch_bins_ptr, cuda_bin_upper_bounds_ptr, cuda_bin_upper_bounds_size, 
                                                          cuda_should_feature_mapped, cuda_batch_value_ptr, cur_cuda_batch_size,
                                                          cuda_feature2group, cuda_feature2subfeature, cuda_groups_is_multi_val, cuda_most_freq_bins,
                                                          cuda_bin_type_is_numerical, cuda_missing_type_is_nan, cuda_categorical_2_bin_ptr);
}
  
}

#endif