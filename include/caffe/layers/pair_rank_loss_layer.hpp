#ifndef CAFFE_PAIR_RANK_LOSS_LAYER_HPP_
#define CAFFE_PAIR_RANK_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class PairRankLossLayer : public LossLayer<Dtype> {
 public:
  explicit PairRankLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
//  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
//      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PairRankLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  
  Dtype margin_;
  Blob<Dtype> ordered_diff_;  // cached for backward pass
  Blob<Dtype> similar_diff_; // cached for similar pairs
};

}  // namespace caffe

#endif  // CAFFE_PAIR_RANK_LOSS_LAYER_HPP_
