#include <cfloat>
#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/sim_cross_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SimCrossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(bottom.size() == 2);
  CHECK(bottom[0]->num() == bottom[1]->num());
  CHECK(bottom[0]->height() == bottom[1]->height());

  dist_mode_ = this->layer_param_.sim_cross_param().dist_mode();
  if(dist_mode_ == 2) {
    const bool bias_term_ = this->layer_param_.sim_cross_param().bias_term();
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    vector<int> weight_shape(3);
    weight_shape[0] = this->layer_param_.sim_cross_param().mesure_count();
    weight_shape[1] = bottom[0]->height();
    weight_shape[2] = bottom[1]->height();
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.sim_cross_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    
    // If necessary, initialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(3);
      bias_shape[0] = this->layer_param_.sim_cross_param().mesure_count();
      bias_shape[1] = bottom[0]->channels();
      bias_shape[2] = bottom[1]->channels();
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.sim_cross_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
}

template <typename Dtype>
void SimCrossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape;
  top_shape.push_back(bottom[0]->num());
  if(dist_mode_== 2) {
    top_shape.push_back(this->layer_param_.sim_cross_param().mesure_count());
  }
  else {
    top_shape.push_back(1);
  }
  top_shape.push_back(bottom[0]->channels());
  top_shape.push_back(bottom[1]->channels());
  top[0]->Reshape(top_shape);

  if(dist_mode_ == 0) { // save data's l2-norm
    vector<int> sz;
    sz.push_back(bottom[0]->num());
    sz.push_back(bottom[0]->channels());
    data0_norm_.Reshape(sz);
    sz[1] = bottom[1]->channels();
    data1_norm_.Reshape(sz);
  }
  else if(dist_mode_ == 2) {
    vector<int> sz;
    sz.push_back(bottom[0]->channels());
    sz.push_back(bottom[0]->height());
    measure_temp0_.Reshape(sz);
    sz[0]=bottom[1]->channels();
    measure_temp1_.Reshape(sz);
  }
}

template <typename Dtype>
void SimCrossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  const Dtype* bottom_data0 = bottom[0]->cpu_data();
  const Dtype* bottom_data1 = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  const int num = bottom[0]->num();
  const int wordnum1 = bottom[0]->channels();
  const int wordnum2 = bottom[1]->channels();

  const int dim = bottom[0]->height();

  if(dist_mode_ == 1) {
    for(int i = 0; i < num; i ++) {
      for(int j = 0; j < wordnum1; j ++) {
        for(int k = 0; k < wordnum2; k ++) {
          Dtype dist = 0;
          for(int dd = 0; dd < dim; dd++) {
            Dtype diff = *(bottom_data0+i*wordnum1*dim+j*dim+dd) - *(bottom_data1+\
                          i*wordnum2*dim+k*dim+dd);
            dist += diff * diff;
          }
          dist = sqrt(dist);
          top_data[i*wordnum1*wordnum2 + j* wordnum2+k] = 1/(1+dist);
        }
      }
    }
  }
  else if(dist_mode_ == 0) {
    Dtype* data0_norm = data0_norm_.mutable_cpu_data();
    Dtype* data1_norm = data1_norm_.mutable_cpu_data();
    // compute data0 norm
    for(int i = 0; i < num; i ++) {
      for(int j = 0; j < wordnum1; j ++) {
          data0_norm[i*wordnum1+j] = sqrt(caffe_cpu_dot(dim, bottom_data0+i*wordnum1*dim+j*dim, \
                                        bottom_data0+i*wordnum1*dim+j*dim));
      }
    }

    // compute data1 norm
    for(int i = 0; i < num; i ++) {
      for(int j = 0; j < wordnum2; j ++) {
          data1_norm[i*wordnum2+j] = sqrt(caffe_cpu_dot(dim, bottom_data1+i*wordnum2*dim+j*dim, \
                                        bottom_data1+i*wordnum2*dim+j*dim));
      }
    }

    for(int i = 0; i < num; i ++) {
      for(int j = 0; j < wordnum1; j ++) {
        for(int k = 0; k < wordnum2; k ++) {
            top_data[i*wordnum1*wordnum2+j*wordnum2+k] = caffe_cpu_dot(dim, bottom_data0+i*wordnum1*dim+j*dim, \
                                        bottom_data1+i*wordnum2*dim+k*dim)/data0_norm[i*wordnum1+j]/data1_norm[i*wordnum2+k];
        }
      }
    }
  }
  else if(dist_mode_ == 2) {
    const int measure_count = this->layer_param_.sim_cross_param().mesure_count();
    const bool bias_term_ = this->layer_param_.sim_cross_param().bias_term();
    const Dtype* weight = this->blobs_[0]->cpu_data();

    Dtype* temp_data = measure_temp0_.mutable_cpu_data();
    for(int i = 0; i < num; i ++) {
      for(int j = 0; j < measure_count; j ++) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, wordnum1, dim, dim, (Dtype)1.,
          bottom_data0+i*wordnum1*dim, weight+j*dim*dim, (Dtype)0., temp_data);
          
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, wordnum1, wordnum2, dim, (Dtype)1.,
          temp_data, bottom_data1+i*wordnum2*dim, (Dtype)0., 
          top_data+i*measure_count*wordnum1*wordnum2+j*wordnum1*wordnum2);
      }
      if(bias_term_) {
        caffe_add(measure_count*wordnum1*wordnum2, this->blobs_[1]->cpu_data(), 
                      top_data+i*measure_count*wordnum1*wordnum2, 
                      top_data+i*measure_count*wordnum1*wordnum2);
      }
    }
  }
  
}

template <typename Dtype>
void SimCrossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, 
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int num = bottom[0]->num();
  int wordnum1 = bottom[0]->channels();
  int wordnum2 = bottom[1]->channels();
  int dim = bottom[0]->height();

  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();  
    
  caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
  caffe_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_cpu_diff());

#if 0
  if(dist_mode_== 2) {
    printf("\n----------------bgn0------------------\n\n\n");
    for(int i = 0; i < bottom[0]->count(); i ++) {
      printf("[%f,%f]  ", bottom[0]->cpu_data()[i], bottom[0]->cpu_diff()[i]);
    }
    printf("\n\n\n");
    for(int i = 0; i < bottom[1]->count(); i ++) {
      printf("[%f,%f]  ", bottom[1]->cpu_data()[i], bottom[1]->cpu_diff()[i]);
    }
    printf("\n\n\n");
    for(int i = 0; i < top[0]->count(); i ++) {
      printf("[%f,%f]  ", top_data[i], top_diff[i]);
    }
    printf("\n\n\n");
    for(int i = 0; i < this->blobs_[0]->count(); i ++) {
      printf("[%f,%f]  ", this->blobs_[0]->cpu_data()[i], this->blobs_[0]->cpu_diff()[i]);
    }
    printf("\n\n\n----------------end0------------------\n");
  }
#endif
  
  if(propagate_down[0] || propagate_down[1]) {
    //printf("\n\n\n*****%d,%d,%d*********\n\n\n", num, wordnum1, wordnum2);
    const Dtype* bottom_data0 = bottom[0]->cpu_data();
    const Dtype* bottom_data1 = bottom[1]->cpu_data();
    Dtype* bottom_diff0 = bottom[0]->mutable_cpu_diff();
    Dtype* bottom_diff1 = bottom[1]->mutable_cpu_diff();
    
    if(dist_mode_ == 1) {
      for(int dd = 0; dd < dim; dd++) {
        for(int j = 0; j < num; j ++) {
          for(int k = 0; k < wordnum1; k ++) {
            for(int m = 0; m < wordnum2; m ++) {
              int t_ind = j*wordnum1*wordnum2+k*wordnum2+m;
              int b_ind0 = j*wordnum1*dim+k*dim+dd;
              int b_ind1 = j*wordnum2*dim+m*dim+dd;
              Dtype tt = top_diff[t_ind]*top_data[t_ind]*top_data[t_ind]*top_data[t_ind]*\
                      (bottom_data0[b_ind0]-bottom_data1[b_ind1])/(top_data[t_ind] - 1+1e-9);
              
              bottom_diff0[b_ind0] +=  tt;                          
              bottom_diff1[b_ind1] += -tt;
            }
          }
        }
      }
    }
    else if(dist_mode_ == 0){
      const Dtype* data0_norm = data0_norm_.mutable_cpu_data();
      const Dtype* data1_norm = data1_norm_.mutable_cpu_data();
      for(int dd = 0; dd < dim; dd++) {
        for(int j = 0; j < num; j ++) {
          for(int k = 0; k < wordnum1; k ++) {
            for(int m = 0; m < wordnum2; m ++) {
              int t_ind = j*wordnum1*wordnum2+k*wordnum2+m;
              int b_ind0 = j*wordnum1*dim+k*dim+dd;
              int b_ind1 = j*wordnum2*dim+m*dim+dd;
              const Dtype nrm0 = data0_norm[j*wordnum1+k];
              const Dtype nrm1 = data1_norm[j*wordnum2+m];
              
              Dtype tt = top_diff[t_ind]*(bottom_data1[b_ind1]/nrm0/nrm1 -\
                  bottom_data0[b_ind0]*top_data[t_ind]/(nrm0*nrm0));
              bottom_diff0[b_ind0] +=  tt; 
              
              tt = top_diff[t_ind]*(bottom_data0[b_ind0]/nrm0/nrm1 -\
                  bottom_data1[b_ind1]*top_data[t_ind]/(nrm1*nrm1));                    
              bottom_diff1[b_ind1] += tt;
            }
          }
        }
      }
    }
    else if(dist_mode_ == 2) {
      const int measure_count = this->layer_param_.sim_cross_param().mesure_count();
      const bool bias_term_ = this->layer_param_.sim_cross_param().bias_term();
      const Dtype* weight = this->blobs_[0]->cpu_data();
      
      caffe_set(this->blobs_[0]->count(), Dtype(0), this->blobs_[0]->mutable_cpu_diff());
      // old ways
      /*for(int i = 0; i < num; i ++) {
        for(int j = 0; j < measure_count; j ++) {
          for(int k = 0; k < wordnum1; k ++) {
            for(int m = 0; m < wordnum2; m ++) {
              const int top_index = i*measure_count*wordnum1*wordnum2+j*wordnum1*wordnum2+k*wordnum2+m;
              const int bottom0_index = i*wordnum1*dim+k*dim;
              const int bottom1_index = i*wordnum2*dim+m*dim;
              const int topdiff_index = j*dim*dim;
              caffe_cpu_ger<Dtype>(dim, dim, top_diff[top_index], bottom_data0+bottom0_index, 
                bottom_data1+bottom1_index, this->blobs_[0]->mutable_cpu_diff()+topdiff_index);
                
              caffe_cpu_gemv<Dtype>(CblasNoTrans, dim, dim, top_diff[top_index], weight+topdiff_index, 
                bottom_data1+bottom1_index, Dtype(1), 
                bottom_diff0+bottom0_index);
                
              caffe_cpu_gemv<Dtype>(CblasTrans, dim, dim, top_diff[top_index], weight+topdiff_index, 
                bottom_data0+bottom0_index, Dtype(1), 
                bottom_diff1+bottom1_index);
            }
          }
        }
      }*/
      Dtype* temp0 = measure_temp0_.mutable_cpu_data();
      Dtype* temp1 = measure_temp1_.mutable_cpu_data();
      for(int i = 0; i < num; i ++) {
        for(int j = 0; j < measure_count; j ++) {
          int topdiff_index = i*measure_count*wordnum1*wordnum2+j*wordnum1*wordnum2;
          
          caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, dim, wordnum2, wordnum1, (Dtype)1.,
            bottom_data0+i*wordnum1*dim, top_diff+topdiff_index, (Dtype)0., temp1);            
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, dim, dim, wordnum2, (Dtype)1.,
            temp1, bottom_data1+i*wordnum2*dim, (Dtype)1., this->blobs_[0]->mutable_cpu_diff()+j*dim*dim);
            
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, dim, wordnum2, dim, (Dtype)1.,
            weight+j*dim*dim, bottom_data1+i*wordnum2*dim, (Dtype)0., temp1);            
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, wordnum1, dim, wordnum2, (Dtype)1.,
            top_diff+topdiff_index, temp1, (Dtype)1., bottom_diff0+i*wordnum1*dim);
            
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, wordnum1, dim, dim, (Dtype)1.,
            bottom_data0+i*wordnum1*dim, weight+j*dim*dim, (Dtype)0., temp0);            
          caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, wordnum2, dim, wordnum1, (Dtype)1.,
            top_diff+topdiff_index, temp0, (Dtype)1., bottom_diff1+i*wordnum2*dim);
        }
        if(bias_term_) { // && this->param_propagate_down_[1]
          caffe_add(measure_count*wordnum1*wordnum2, top_diff+i*measure_count*wordnum1*wordnum2, 
                  this->blobs_[1]->cpu_diff(), this->blobs_[1]->mutable_cpu_diff());
        }
      }
      /**/
    }
  }// end if

#if 0
  if(dist_mode_ == 2) {
    int k = propagate_down[0] == 1 ? 0 : 1; 
    printf("\n\n\n----------------bgn(%d)------------------\n\n\n", k);
    for(int i = 0; i < bottom[k]->count(); i ++) {
      printf("[%f,%f]  ", bottom[k]->cpu_data()[i], bottom[k]->cpu_diff()[i]);
    }
    printf("\n\n\n");
    for(int i = 0; i < top[0]->count(); i ++) {
      printf("[%f,%f]  ", top[0]->cpu_data()[i], top[0]->cpu_diff()[i]);
    }  
    printf("\n\n\n");
    for(int i = 0; i < data0_norm_.count(); i ++) {
      printf("[%f]  ", data0_norm_.cpu_data()[i]);
    }
    printf("\n\n\n");
    for(int i = 0; i < this->blobs_[0]->count(); i ++) {
      printf("[%f,%f]  ", this->blobs_[0]->cpu_data()[i], this->blobs_[0]->cpu_diff()[i]);
    }
    printf("\n\n\n----------------end(%d)------------------\n\n\n", k);
  }
  //exit(0);
#endif
  
}

#ifdef CPU_ONLY
STUB_GPU(SimCrossLayer);
#endif

INSTANTIATE_CLASS(SimCrossLayer);
REGISTER_LAYER_CLASS(SimCross);

}  // namespace caffe
