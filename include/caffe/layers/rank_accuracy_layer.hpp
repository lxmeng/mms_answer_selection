#ifndef CAFFE_RANK_ACCURACY_LAYER_HPP_
#define CAFFE_RANK_ACCURACY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class RankAccuracyLayer : public Layer<Dtype> {
 public:
  explicit RankAccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RankAccuracy"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }

  // If there are two top blobs, then the second blob will contain
  // accuracies per class.
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlos() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /// @brief Not implemented -- AccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

};

}  // namespace caffe

#endif  // CAFFE_RANK_ACCURACY_LAYER_HPP_
