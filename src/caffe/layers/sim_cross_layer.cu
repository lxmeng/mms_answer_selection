#include <cfloat>
#include <vector>

#include "caffe/layers/sim_cross_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/util/gpu_util.cuh"

namespace caffe {

template <typename Dtype>
__global__ void simEucForward(const int nthreads, const Dtype* data1,
    const Dtype* data2, const int wordnum1, const int wordnum2, 
    const int dim, const int dim_ind, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int s_ind = index/(wordnum1*wordnum2);
    const int h_ind = (index-s_ind*wordnum1*wordnum2)/wordnum2;//make sure wordnum2 > wordnum1
    const int v_ind = index-s_ind*wordnum1*wordnum2-h_ind*wordnum2;
    Dtype diff = data1[s_ind*wordnum1*dim+h_ind*dim+dim_ind] - \
                 data2[s_ind*wordnum2*dim+v_ind*dim+dim_ind];
    top_data[index] += diff * diff;
  }
}

template <typename Dtype>
__global__ void simTrans(const int nthreads, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    top_data[index] = 1/(1+sqrt(top_data[index]));
  }
}

template <typename Dtype>
__global__ void l2_norm_add(const int nthreads, const int wordnum, 
  const int dim, const Dtype* data, Dtype* data_nrm) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n_ind = index/(dim*wordnum);
    const int w_ind = (index-n_ind*dim*wordnum)/dim;
    caffe_gpu_atomic_add(data[index]*data[index], data_nrm+n_ind*wordnum+w_ind);
  }
}

template <typename Dtype>
__global__ void CosSimForward(const int nthreads, const Dtype* data0,
    const Dtype* data1, const Dtype* data0_nrm, const Dtype* data1_nrm, 
    const int wordnum1, const int wordnum2, 
    const int dim, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n_ind = index/(wordnum1*wordnum2*dim);
    int t = index - n_ind*wordnum1*wordnum2*dim;
    const int w0_ind = t/(wordnum2*dim);
    t -= w0_ind*wordnum2*dim;
    const int w1_ind = t/dim;
    const int d_ind = t - w1_ind*dim;
    
    const int b_ind0 = n_ind*wordnum1*dim+w0_ind*dim+d_ind;
    const int b_ind1 = n_ind*wordnum2*dim+w1_ind*dim+d_ind;
    const int b_r_ind0 = n_ind*wordnum1+w0_ind;
    const int b_r_ind1 = n_ind*wordnum2+w1_ind;
    const int t_ind = n_ind*wordnum1*wordnum2+w0_ind*wordnum2+w1_ind;
    
    const Dtype rst = data0[b_ind0]*data1[b_ind1]/sqrt(data0_nrm[b_r_ind0])/sqrt(data1_nrm[b_r_ind1]);
    caffe_gpu_atomic_add(rst, top_data+t_ind);
  }
}

template <typename Dtype>
__global__ void simEucBackward(const int nthreads, const Dtype* top_data,
    const Dtype* top_diff, const Dtype* btm_data0, const Dtype* btm_data1, 
    const int wordnum1, const int wordnum2, const int dim, const int dim_ind, 
    Dtype* btm_diff0, Dtype* btm_diff1) { //, Dtype* btm_diff0, Dtype* btm_diff1
    
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int s_ind = index/(wordnum1*wordnum2);
    const int h_ind = (index-s_ind*wordnum1*wordnum2)/wordnum2;//make sure wordnum2 > wordnum1
    const int v_ind = index-s_ind*wordnum1*wordnum2-h_ind*wordnum2;

    const int b_ind0 = s_ind*wordnum1*dim+h_ind*dim+dim_ind;
    const int b_ind1 = s_ind*wordnum2*dim+v_ind*dim+dim_ind;
    
    const Dtype tt = top_diff[index]*top_data[index]*top_data[index]*top_data[index]*\
                    (btm_data0[b_ind0]-btm_data1[b_ind1])/(top_data[index]-1+1e-9);
         
    //btm_diff0[b_ind0] += tt;
    //btm_diff1[b_ind1] += -tt;
    caffe_gpu_atomic_add(tt, btm_diff0+b_ind0);
    caffe_gpu_atomic_add(-tt, btm_diff1+b_ind1);      
    
    //printf("\n[%d, %d, %d, %d, %d, %d, %f, %f]\n",index, s_ind, h_ind, v_ind, b_ind0, b_ind1, tt, btm_diff0[b_ind0]);         
  }
  
}

template <typename Dtype>
__global__ void CosSimBackward(const int nthreads, const Dtype* top_data,
    const Dtype* top_diff, const Dtype* btm_data0, const Dtype* btm_data1, 
    const Dtype* btm_data0_nrm_sqr, const Dtype* btm_data1_nrm_sqr,
    const int wordnum1, const int wordnum2, const int dim,
    Dtype* btm_diff0, Dtype* btm_diff1) {
    
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n_ind = index/(wordnum1*wordnum2*dim);
    int t = index - n_ind*wordnum1*wordnum2*dim;
    const int w0_ind = t/(wordnum2*dim);
    t = t - w0_ind*wordnum2*dim;
    const int w1_ind = t/dim;
    const int d_ind = t - w1_ind*dim;
    
    const int b_ind0 = n_ind*wordnum1*dim+w0_ind*dim+d_ind;
    const int b_ind1 = n_ind*wordnum2*dim+w1_ind*dim+d_ind;
    const int b_r_ind0 = n_ind*wordnum1+w0_ind;
    const int b_r_ind1 = n_ind*wordnum2+w1_ind;
    const int t_ind = n_ind*wordnum1*wordnum2+w0_ind*wordnum2+w1_ind;
    
    const Dtype nrm0 = sqrt(btm_data0_nrm_sqr[b_r_ind0]);
    const Dtype nrm1 = sqrt(btm_data1_nrm_sqr[b_r_ind1]);
    
    Dtype tt = top_diff[t_ind]*(btm_data1[b_ind1]/nrm0/nrm1 -\
                  btm_data0[b_ind0]*top_data[t_ind]/(nrm0*nrm0));
    caffe_gpu_atomic_add(tt, btm_diff0+b_ind0);
    
    tt = top_diff[t_ind]*(btm_data0[b_ind0]/nrm0/nrm1 -\
                  btm_data1[b_ind1]*top_data[t_ind]/(nrm1*nrm1));     
    caffe_gpu_atomic_add(tt, btm_diff1+b_ind1);              
  } 
}

template <typename Dtype>
void SimCrossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    
#if 0
  Forward_cpu(bottom, top);
#else  

  const int count = top[0]->count();
  const int dim = bottom[0]->height();
  
  const int wordnum1 = bottom[0]->channels();
  const int wordnum2 = bottom[1]->channels();

  //Dtype* tpdata = top[0]->mutable_cpu_data();
  caffe_set(count, Dtype(0), top[0]->mutable_cpu_data());
  
  const Dtype* btm_data0 = bottom[0]->gpu_data();
  const Dtype* btm_data1 = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  
  if(dist_mode_ ==  1) {
    for(int i = 0; i < dim; i ++) {
      simEucForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, btm_data0, btm_data1, wordnum1, wordnum2, dim, i,
            top_data);
    }
  
    //caffe_gpu_powx(count, top[0]->gpu_data(), Dtype(0.5), top[0]->mutable_gpu_data());
    //caffe_gpu_add_scalar(count, Dtype(1), top[0]->mutable_gpu_data());
    //caffe_gpu_powx(count, top[0]->gpu_data(), Dtype(-1), top[0]->mutable_gpu_data());
    
    simTrans<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_data);
  }
  else if(dist_mode_ ==  0) {
    
    caffe_set(data0_norm_.count(), Dtype(0), data0_norm_.mutable_cpu_data());
    caffe_set(data1_norm_.count(), Dtype(0), data1_norm_.mutable_cpu_data());
    
    Dtype* data0_norm_squar = data0_norm_.mutable_gpu_data();
    Dtype* data1_norm_squar = data1_norm_.mutable_gpu_data();
    
    l2_norm_add<Dtype> <<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
            bottom[0]->count(), wordnum1, dim, btm_data0, data0_norm_squar);
    //caffe_gpu_powx(data0_norm_.count(), data0_norm, Dtype(2), data0_norm);
    
    l2_norm_add<Dtype> <<<CAFFE_GET_BLOCKS(bottom[1]->count()), CAFFE_CUDA_NUM_THREADS>>>(
            bottom[1]->count(), wordnum2, dim, btm_data1, data1_norm_squar);
    //caffe_gpu_powx(data1_norm_.count(), data1_norm_.gpu_data(), Dtype(0.5), data1_norm);
    
    const int count_all = bottom[0]->num()*wordnum1*wordnum2*dim;
    
    CosSimForward<Dtype> <<<CAFFE_GET_BLOCKS(count_all), CAFFE_CUDA_NUM_THREADS>>>(
        count_all, btm_data0, btm_data1, data0_norm_.gpu_data(),
        data1_norm_.gpu_data(), wordnum1, wordnum2, dim,
        top_data);
        
    //LOG(INFO) << bottom[0]->cpu_data()[601] << "," << bottom[0]->cpu_data()[701] << "," << bottom[1]->cpu_data()[1801] << "," << bottom[1]->cpu_data()[2010] << "," << top[0]->cpu_data()[624] << "," << top[0]->cpu_data()[800];
  }
  else if(dist_mode_ == 2) {
    Forward_cpu(bottom, top);
  }
          
  
#endif

}

template <typename Dtype>
void SimCrossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
#if 0
  Backward_cpu(top, propagate_down, bottom);
#else
  const int num = bottom[0]->num();
  const int count = top[0]->count();
  const int dim = bottom[0]->height();
  const int wordnum1 = bottom[0]->channels();
  const int wordnum2 = bottom[1]->channels();
  
  const Dtype* top_data =  top[0]->gpu_data();
  const Dtype* top_diff =  top[0]->gpu_diff();
  
  caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
  caffe_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_cpu_diff());
  
  if(propagate_down[0] || propagate_down[1]) {
    const Dtype* bottom_data0 =  bottom[0]->gpu_data();
    const Dtype* bottom_data1 =  bottom[1]->gpu_data();
  
    Dtype* bottom_diff0 =  bottom[0]->mutable_gpu_diff();
    Dtype* bottom_diff1 =  bottom[1]->mutable_gpu_diff();
    
    if(dist_mode_ == 1) {
      for(int i = 0; i < dim; i ++) {    
        simEucBackward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
              count, top_data, top_diff, 
              bottom_data0, bottom_data1, 
              wordnum1, wordnum2, dim, i,
              bottom_diff0, bottom_diff1);       
      }
    }
    else if(dist_mode_ == 0) {
      const int count_all = num*wordnum1*wordnum2*dim;
      CosSimBackward<Dtype> <<<CAFFE_GET_BLOCKS(count_all), CAFFE_CUDA_NUM_THREADS>>>(
          count_all, top_data, top_diff, 
          bottom_data0, bottom_data1,
          data0_norm_.gpu_data(),
          data1_norm_.gpu_data(), 
          wordnum1, wordnum2, dim,
          bottom_diff0, bottom_diff1); 
    }
    else if(dist_mode_ == 2) {
      Backward_cpu(top, propagate_down, bottom);
    }
  }
    
#if 0
  if(dist_mode_ == 0) {
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
    printf("\n\n\n----------------end(%d)------------------\n\n\n", k);
    //exit(0);
  }
#endif

#endif
 
}

INSTANTIATE_LAYER_GPU_FUNCS(SimCrossLayer);

}  // namespace caffe
