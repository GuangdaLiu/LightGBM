/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_NEW_CUDA_TREE_LEARNER_HPP_
#define LIGHTGBM_NEW_CUDA_TREE_LEARNER_HPP_

#ifdef USE_CUDA

#include "../serial_tree_learner.h"
#include "cuda_leaf_splits.hpp"
#include "cuda_histogram_constructor.hpp"
#include "cuda_data_partition.hpp"
#include "cuda_best_split_finder.hpp"
#include "cuda_centralized_info.hpp"
#include "cuda_score_updater.hpp"
#include "cuda_binary_objective.hpp"
#include "cuda_regression_objective.hpp"
#include "cuda_ranking_objective.hpp"

namespace LightGBM {

class NewCUDATreeLearner: public SerialTreeLearner {
 public:
  explicit NewCUDATreeLearner(const Config* config);

  ~NewCUDATreeLearner();

  void Init(const Dataset* train_data, bool is_constant_hessian) override;

  void ResetTrainingData(const Dataset* train_data,
                         bool is_constant_hessian) override;

  Tree* Train(const score_t* gradients, const score_t *hessians, bool is_first_tree) override;
  
  void SetBaggingData(const Dataset* subset, const data_size_t* used_indices, data_size_t num_data) override;

  void AddPredictionToScore(const Tree* tree, double* out_score) const override;

 protected:
  void FindBestSplits(const Tree* tree) override;

  void ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) override;

  void FindBestSplitsFromHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract, const Tree* tree) override;

  void Split(Tree* tree, int best_leaf, int* left_leaf, int* right_leaf) override;

  void BeforeTrain() override;

  // number of GPUs
  int num_gpus_;
  // number of threads on CPU
  int num_threads_;

  // CUDA components for tree training
  // centralized information shared by other CUDA components
  std::unique_ptr<CUDACentralizedInfo> cuda_centralized_info_;
  // leaf splits information for smaller and larger leaves
  std::unique_ptr<CUDALeafSplits> cuda_smaller_leaf_splits_;
  std::unique_ptr<CUDALeafSplits> cuda_larger_leaf_splits_;
  // data partition that partitions data indices into different leaves
  std::unique_ptr<CUDADataPartition> cuda_data_partition_;
  // for histogram construction
  std::unique_ptr<CUDAHistogramConstructor> cuda_histogram_constructor_;
  // for best split information finding, given the histograms
  std::unique_ptr<CUDABestSplitFinder> cuda_best_split_finder_;

  std::vector<int> leaf_best_split_feature_;
  std::vector<uint32_t> leaf_best_split_threshold_;
  std::vector<uint8_t> leaf_best_split_default_left_;
  std::vector<data_size_t> leaf_num_data_;
  std::vector<data_size_t> leaf_data_start_;
  std::vector<double> leaf_sum_hessians_;
  int smaller_leaf_index_;
  int larger_leaf_index_;
  int best_leaf_index_;
};

}  // namespace LightGBM

#else  // USE_CUDA

// When GPU support is not compiled in, quit with an error message

namespace LightGBM {

class NewCUDATreeLearner: public SerialTreeLearner {
 public:
    #pragma warning(disable : 4702)
    explicit NewCUDATreeLearner(const Config* tree_config) : SerialTreeLearner(tree_config) {
      Log::Fatal("CUDA Tree Learner was not enabled in this build.\n"
                 "Please recompile with CMake option -DUSE_CUDA=1");
    }
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_NEW_CUDA_TREE_LEARNER_HPP_
