#ifndef LIGHTGBM_SAMPLE_STRATEGY_BAGGING_HPP_
#define LIGHTGBM_SAMPLE_STRATEGY_BAGGING_HPP_

namespace LightGBM {

class BAGGING : public SampleStrategy {
 public:
  BAGGING(const Config* config, const Dataset* train_data, const ObjectiveFunction* objective_function, int num_tree_per_iteration) 
    : need_re_bagging_(false) {
    config_ = config;
    train_data_ = train_data;
    num_data_ = train_data->num_data();
    objective_function_ = objective_function;
    num_tree_per_iteration_ = num_tree_per_iteration;
  }
  ~BAGGING() {}

  void Bagging(int iter, const Dataset* train_data, TreeLearner* tree_learner, score_t* gradients, score_t* hessians) override {
    Common::FunctionTimer fun_timer("GBDT::Bagging", global_timer);
    // if need bagging
    if ((bag_data_cnt_ < num_data_ && iter % config_->bagging_freq == 0) ||
        need_re_bagging_) {
      need_re_bagging_ = false;
      auto left_cnt = bagging_runner_.Run<true>(
          num_data_,
          [=](int, data_size_t cur_start, data_size_t cur_cnt, data_size_t* left,
              data_size_t*) {
            data_size_t cur_left_count = 0;
            if (balanced_bagging_) {
              cur_left_count =
                  BalancedBaggingHelper(cur_start, cur_cnt, left);
            } else {
              cur_left_count = BaggingHelper(cur_start, cur_cnt, left);
            }
            return cur_left_count;
          },
          bag_data_indices_.data());
      bag_data_cnt_ = left_cnt;
      Log::Debug("Re-bagging, using %d data to train", bag_data_cnt_);
      // set bagging data to tree learner
      if (!is_use_subset_) {
        tree_learner->SetBaggingData(nullptr, bag_data_indices_.data(), bag_data_cnt_);
      } else {
        // get subset
        tmp_subset_->ReSize(bag_data_cnt_);
        tmp_subset_->CopySubrow(train_data_, bag_data_indices_.data(),
                                bag_data_cnt_, false);
        tree_learner->SetBaggingData(tmp_subset_.get(), bag_data_indices_.data(),
                                      bag_data_cnt_);
      }
    }
  }

  void ResetGOSS() override {} 

  void ResetBaggingConfig(const Config* config, bool is_change_dataset, 
          std::vector<score_t, Common::AlignmentAllocator<score_t, kAlignedSize>>& gradients, 
          std::vector<score_t, Common::AlignmentAllocator<score_t, kAlignedSize>>& hessians) override {
    // if need bagging, create buffer
    data_size_t num_pos_data = 0;
    if (objective_function_ != nullptr) {
      num_pos_data = objective_function_->NumPositiveData();
    }
    bool balance_bagging_cond = (config->pos_bagging_fraction < 1.0 || config->neg_bagging_fraction < 1.0) && (num_pos_data > 0);
    if ((config->bagging_fraction < 1.0 || balance_bagging_cond) && config->bagging_freq > 0) {
      need_re_bagging_ = false;
      if (!is_change_dataset &&
        config_ != nullptr && config_->bagging_fraction == config->bagging_fraction && config_->bagging_freq == config->bagging_freq
        && config_->pos_bagging_fraction == config->pos_bagging_fraction && config_->neg_bagging_fraction == config->neg_bagging_fraction) {
        return;
      }
      if (balance_bagging_cond) {
        balanced_bagging_ = true;
        bag_data_cnt_ = static_cast<data_size_t>(num_pos_data * config->pos_bagging_fraction)
                        + static_cast<data_size_t>((num_data_ - num_pos_data) * config->neg_bagging_fraction);
      } else {
        bag_data_cnt_ = static_cast<data_size_t>(config->bagging_fraction * num_data_);
      }
      bag_data_indices_.resize(num_data_);
      bagging_runner_.ReSize(num_data_);
      bagging_rands_.clear();
      for (int i = 0;
          i < (num_data_ + bagging_rand_block_ - 1) / bagging_rand_block_; ++i) {
        bagging_rands_.emplace_back(config_->bagging_seed + i);
      }

      double average_bag_rate =
          (static_cast<double>(bag_data_cnt_) / num_data_) / config->bagging_freq;
      is_use_subset_ = false;
      const int group_threshold_usesubset = 100;
      if (average_bag_rate <= 0.5
          && (train_data_->num_feature_groups() < group_threshold_usesubset)) {
        if (tmp_subset_ == nullptr || is_change_dataset) {
          tmp_subset_.reset(new Dataset(bag_data_cnt_));
          tmp_subset_->CopyFeatureMapperFrom(train_data_);
        }
        is_use_subset_ = true;
        Log::Debug("Use subset for bagging");
      }

      need_re_bagging_ = true;

      if (is_use_subset_ && bag_data_cnt_ < num_data_) {
        if (objective_function_ == nullptr) {
          size_t total_size = static_cast<size_t>(num_data_) * num_tree_per_iteration_;
          gradients.resize(total_size);
          hessians.resize(total_size); 
        }
      }
    } else {
      bag_data_cnt_ = num_data_;
      bag_data_indices_.clear();
      bagging_runner_.ReSize(0);
      is_use_subset_ = false;
    }
  }

  data_size_t BaggingHelper(data_size_t start, data_size_t cnt, data_size_t* buffer) {
    if (cnt <= 0) {
      return 0;
    }
    data_size_t cur_left_cnt = 0;
    data_size_t cur_right_pos = cnt;
    // random bagging, minimal unit is one record
    for (data_size_t i = 0; i < cnt; ++i) {
      auto cur_idx = start + i;
      if (bagging_rands_[cur_idx / bagging_rand_block_].NextFloat() < config_->bagging_fraction) {
        buffer[cur_left_cnt++] = cur_idx;
      } else {
        buffer[--cur_right_pos] = cur_idx;
      }
    }
    return cur_left_cnt;
  }

  data_size_t BalancedBaggingHelper(data_size_t start, data_size_t cnt, data_size_t* buffer) {
    if (cnt <= 0) {
      return 0;
    }
    auto label_ptr = train_data_->metadata().label();
    data_size_t cur_left_cnt = 0;
    data_size_t cur_right_pos = cnt;
    // random bagging, minimal unit is one record
    for (data_size_t i = 0; i < cnt; ++i) {
      auto cur_idx = start + i;
      bool is_pos = label_ptr[start + i] > 0;
      bool is_in_bag = false;
      if (is_pos) {
        is_in_bag = bagging_rands_[cur_idx / bagging_rand_block_].NextFloat() <
                    config_->pos_bagging_fraction;
      } else {
        is_in_bag = bagging_rands_[cur_idx / bagging_rand_block_].NextFloat() <
                    config_->neg_bagging_fraction;
      }
      if (is_in_bag) {
        buffer[cur_left_cnt++] = cur_idx;
      } else {
        buffer[--cur_right_pos] = cur_idx;
      }
    }
    return cur_left_cnt;
  }

  bool need_re_bagging_;
};

} // namespace LightGBM

#endif  // LIGHTGBM_SAMPLE_STRATEGY_BAGGING_HPP_