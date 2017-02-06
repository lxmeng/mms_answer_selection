#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/auc_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AUCLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  fixed_axis_ = this->layer_param_.auc_param().fixed_axis();

  has_ignore_label_ =
    this->layer_param_.auc_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.auc_param().ignore_label();
  }
}

template <typename Dtype>
void AUCLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(fixed_axis_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.auc_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

//template <typename Dtype>
bool mycompare_auc(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs) {
    return lhs.first > rhs.first;
}

template <typename Dtype>
void AUCLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype auc_value = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  //vector<Dtype> maxval(top_k_+1);
  //vector<int> max_id(top_k_+1);
  int high = 0;
  int count = 0;
  
  //LOG(INFO) << "AUC_CHECK(inner_num:" << inner_num_ << ", outer_num:" << outer_num_ << ",fixed_axis:" << fixed_axis_ << ")";
  
  std::vector<std::pair<Dtype, int> > bottom_data_vector;
  bottom_data_vector.clear();
  
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);
      
	  //get_pair
      bottom_data_vector.push_back(std::make_pair(
            bottom_data[i * dim + fixed_axis_ * inner_num_ + j], label_value));
			
	  ++ count;
    }
  }
  
  //LOG(INFO) << "size:" << bottom_data_vector.size();
  //for(int i = 0; i < bottom_data_vector.size(); i ++)
  //{
  //  LOG(INFO) << "(" << bottom_data_vector[i].first << ") ";
  //}
  //2015-07-31 MLX-TEST
  // for(int i = 0; i < 10; i ++)
 // {
  //  LOG(INFO) << bottom_data_vector[i].second << ", " << bottom_data_vector[i].first;
 // }
  
  std::sort(
	  bottom_data_vector.begin(), bottom_data_vector.end(),
	  mycompare_auc); //std::greater<std::pair<Dtype, int> >());
     
  //2015-07-24 TEST
  /*for(int i = 0; i < bottom_data_vector.size()-1; i ++)
  {
    Dtype t1 = bottom_data_vector[i].first;
    Dtype t2 = bottom_data_vector[i+1].first;
    if(t1 < t2)
    {
      printf("!!!!!!!!!!!!!!!!!!!!!!!!!!bug1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    }
  }
  
  for(int i = 0; i < bottom_data_vector.size(); i ++)
  {
    int t1 = bottom_data_vector[i].second;
    if(t1 != 0 && t1 != 1)
    {
      printf("!!!!!!!!!!!!!!!!!!!!!!!!!!bug2!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    }
  }*/
     
  //LOG(INFO) << "v1(" << bottom_data_vector[0].first << ") v2(" << bottom_data_vector[1].first << ") relation : " << (bottom_data_vector[0].first<bottom_data_vector[1].first);
  
  for(int i = 0; i < bottom_data_vector.size(); i ++)
  {
	  high += bottom_data_vector[i].second;
	  auc_value += high * (1 - bottom_data_vector[i].second);
  }
  
  //LOG(INFO) << "high:" << high << " Count:" << count << "," << bottom_data_vector.size() << " auc_value:" << auc_value / (high * (count - high));
  //LOG(INFO) << "AUC: " << auc_value / (high * (count - high));
  if(high > 0)
  {
    top[0]->mutable_cpu_data()[0] = auc_value / high / (count - high);
  }
  else
  {
    top[0]->mutable_cpu_data()[0] = 0;
  }
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(AUCLayer);
REGISTER_LAYER_CLASS(AUC);

}  // namespace caffe
