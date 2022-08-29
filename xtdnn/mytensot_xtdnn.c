#include <sys/time.h>
#include "csi_nn.h"
#include "csi_internal.h"

#define FALSE 0
#define TRUE 1
#define IN mid_results_array[mid_results_count]
#define CURRENT mid_results_array[mid_results_count]
#define OUT mid_results_array[++mid_results_count]
#define KERNEL mid_need_params[mid_need_params_count++]
#define WEIGHT mid_need_params[mid_need_params_count++]
#define BIAS mid_need_params[mid_need_params_count++]

#define INIT_IN input()
#define INIT_OUT mid_results_array[mid_results_count+1]

#define MAX_LENGTH 100

//#define R_DEBUG
//#define R_DEBUG_ALL
//#define R_DEBUG_CONV
//#define NOT_RUN

#define LAYOUT CSINN_NHWC
#define DTYPE CSINN_DTYPE_INT8


int run_status = 0;

struct csi_tensor *mid_results_array[MAX_LENGTH];
struct csi_tensor *mid_need_params[MAX_LENGTH];
void *mid_params_config[MAX_LENGTH];

char mid_results_str[MAX_LENGTH][255] = { 0 };


int mid_results_count = 0;	//current pointer of stack
int mid_need_params_count = 0;
int mid_params_config_count = 0;

int all_mid_results_count = 0; //length of stack
int all_mid_need_params_count = 0;
int all_mid_params_config_count = 0;

void random_fill_tensor(struct csi_tensor *tensor){
	int size = 1;
	for(int s = 0; s < 4; s++){
		size *= (tensor->dim)[s];
	}
	for(int i = 0; i < size; i++){
		((char *)(tensor->data))[i] = (char)0;//rand();
	}
}

struct csi_tensor *make_tensor_nchw(int n,int c,int h,int w,int dtype){
	struct csi_tensor *tensor = csi_alloc_tensor(NULL);
	int size = n*c*h*w;
	int elem_size = 1;
	if (dtype == CSINN_DTYPE_INT32) elem_size = 4;
	tensor->data = calloc(elem_size,size);
	tensor->dtype = dtype;
	(tensor->dim)[0] = n;
	(tensor->dim)[1] = h;
	(tensor->dim)[2] = w;
	(tensor->dim)[3] = c;
	tensor->dim_count = 4;
	tensor->layout = LAYOUT;
	return tensor;
}
void add_mid_result_nchw(int n,int c,int h,int w,const char*name){
	mid_results_array[all_mid_results_count++] = make_tensor_nchw(n,c,h,w,DTYPE);
	strcpy(mid_results_str[all_mid_results_count - 1],name);
}

void add_mid_result_nchw_i32(int n,int c,int h,int w,const char*name){
	mid_results_array[all_mid_results_count++] = make_tensor_nchw(n,c,h,w,CSINN_DTYPE_INT32);
	strcpy(mid_results_str[all_mid_results_count - 1],name);
}
void add_compute_params_nchw(int n,int c,int h,int w,const char*name){
	mid_need_params[all_mid_need_params_count++] = make_tensor_nchw(n,c,h,w,DTYPE);
	random_fill_tensor(mid_need_params[all_mid_need_params_count-1]);
}
void add_compute_params_nchw_i32(int n,int c,int h,int w,const char*name){
	mid_need_params[all_mid_need_params_count++] = make_tensor_nchw(n,c,h,w,DTYPE);
	random_fill_tensor(mid_need_params[all_mid_need_params_count-1]);
}

void reset_all_counters(){
	mid_results_count = 0;
	mid_need_params_count = 0;
	mid_params_config_count = 0;
}

struct conv2d_params *conv_param(int stride, int pad){
	if (run_status == 0){
		struct conv2d_params *param = (struct conv2d_params*)calloc(1, sizeof(struct conv2d_params));
		//conv_params[conv_params_count++] = param;
		param->layout = LAYOUT;
		param->stride_height = stride;
		param->stride_width = stride;
		param->pad_top = pad;
		param->pad_left = pad;
		param->pad_down = pad;
		param->pad_right = pad;
		param->dilation_width = 1;
		param->dilation_height = 1;
		param->api = 2;
		param->group = 1;
		mid_params_config[all_mid_params_config_count++] = (void *)param;
		return param;
	} else {
		return (struct conv2d_params *)mid_params_config[mid_params_config_count++];
	}
}

struct diso_params *add_param(){
	if (run_status == 0){
		struct diso_params *param = (struct diso_params *)calloc(1, sizeof(struct diso_params));
		param->layout = LAYOUT;
		param->api = 2;
		mid_params_config[all_mid_params_config_count++] = (void *)param;
		return param;
	} else {
		return (struct diso_params *)mid_params_config[mid_params_config_count++];
	}
}

struct pool_params *pool_param(int size, int stride, int pad){
	if (run_status == 0){
		struct pool_params *param = (struct pool_params *)calloc(1, sizeof(struct pool_params));
		param->pad_top = pad;
		param->pad_left = pad;
		param->stride_width = stride;
		param->stride_height = stride;
		param->filter_width = size;
		param->filter_height = size;
		param->api = 2;
		mid_params_config[all_mid_params_config_count++] = (void *)param;
		return param;
	} else {
		return (struct pool_params *)mid_params_config[mid_params_config_count++];
	}
}

struct relu_params *relu_param(){
	if (run_status == 0){
		struct relu_params *param = (struct relu_params *)calloc(1, sizeof(struct relu_params));
		param->api = 2;
		mid_params_config[all_mid_params_config_count++] = (void *)param;
		return param;
	} else {
		return (struct relu_params *)mid_params_config[mid_params_config_count++];
	}
}

struct fc_params *fc_param(){
	if (run_status == 0){
		struct fc_params *param = (struct fc_params *)calloc(1, sizeof(struct fc_params));
		param->api = 2;
		mid_params_config[all_mid_params_config_count++] = (void *)param;
		return param;
	} else {
		return (struct fc_params *)mid_params_config[mid_params_config_count++];
	}
}

void show_current_layer(const char* layer){
	printf("%d/%d ------------------------------------ [%s] [%s]\n", mid_results_count, all_mid_results_count, layer, mid_results_str[mid_results_count]);
}

void conv2d(struct csi_tensor* in, struct csi_tensor* out, struct csi_tensor* kernel, struct csi_tensor* bias, struct conv2d_params *param){
	if(run_status == 0){
	#ifdef R_DEBUG_ALL
		show_current_layer("conv");
		printf("\tin: %d %d %d %d out: %d %d %d %d\n",in->dim[0],in->dim[1],in->dim[2],in->dim[3],out->dim[0],out->dim[1],out->dim[2],out->dim[3]);
		printf("\tkernel: %d %d %d %d bias: %d %d %d %d\n",kernel->dim[0],kernel->dim[1],kernel->dim[2],kernel->dim[3],bias->dim[0],bias->dim[1],bias->dim[2],bias->dim[3]);
		printf("\tS=%d, P=%d\n",param->stride_width,param->pad_top);
	#endif
		return;
	} else {
	#ifdef R_DEBUG_CONV
		show_current_layer("conv");
		printf("\tin: %d %d %d %d out: %d %d %d %d\n",in->dim[0],in->dim[1],in->dim[2],in->dim[3],out->dim[0],out->dim[1],out->dim[2],out->dim[3]);
		printf("\tkernel: %d %d %d %d bias: %d %d %d %d\n",kernel->dim[0],kernel->dim[1],kernel->dim[2],kernel->dim[3],bias->dim[0],bias->dim[1],bias->dim[2],bias->dim[3]);
		printf("\tS=%d, P=%d\n",param->stride_width,param->pad_top);
	#endif

	}
#ifndef NOT_RUN
#if 1
	if(csi_conv2d_init(in, out, kernel, bias, param) == CSINN_TRUE){
        	//struct timeval new1,new2;
        	//gettimeofday (&new1, NULL);
		//for(int k=0;k<100;k++)
			csi_conv2d(in, out, kernel, bias, param);
        	//gettimeofday (&new2, NULL);
        	//printf("\tConv, time used: %f second \n\n",(new2.tv_sec-new1.tv_sec)+(new2.tv_usec-new1.tv_usec)/1000000.0);
	}
#endif
#endif
}

void relu(struct csi_tensor* in, struct csi_tensor* out, struct relu_params *param){
	if(run_status == 0){
	#ifdef R_DEBUG_ALL
		show_current_layer("relu");
		printf("\tin: %d %d %d %d out: %d %d %d %d\n",in->dim[0],in->dim[1],in->dim[2],in->dim[3],out->dim[0],out->dim[1],out->dim[2],out->dim[3]);
	#endif	
		return;
	} else {
	#ifdef R_DEBUG
		show_current_layer("relu");
		printf("\tin: %d %d %d %d out: %d %d %d %d\n",in->dim[0],in->dim[1],in->dim[2],in->dim[3],out->dim[0],out->dim[1],out->dim[2],out->dim[3]);
	#endif	
	}
#ifndef NOT_RUN
	if(csi_relu_init(in, out, param) == CSINN_TRUE)
		csi_relu(in, out, param);
#endif
}

void add(struct csi_tensor* a, struct csi_tensor* b, struct csi_tensor* out, struct diso_params* param){
	if(run_status == 0){
	#ifdef R_DEBUG_ALL
		show_current_layer("add");
		printf("\ta: %d %d %d %d, b: %d %d %d %d, out: %d %d %d %d\n",a->dim[0],a->dim[1],a->dim[2],a->dim[3],b->dim[0],b->dim[1],b->dim[2],b->dim[3],out->dim[0],out->dim[1],out->dim[2],out->dim[3]);
	#endif	
		return;
	} else {
	#ifdef R_DEBUG
		show_current_layer("add");
		printf("\ta: %d %d %d %d, b: %d %d %d %d, out: %d %d %d %d\n",a->dim[0],a->dim[1],a->dim[2],a->dim[3],b->dim[0],b->dim[1],b->dim[2],b->dim[3],out->dim[0],out->dim[1],out->dim[2],out->dim[3]);
	#endif	
	}
#ifndef NOT_RUN
	if(csi_add_init(a, b, out, param) == CSINN_TRUE)
		csi_add(a, b, out, param);
#endif
}

void maxpool(struct csi_tensor* in, struct csi_tensor* out, struct pool_params *param){
	if(run_status == 0){
	#ifdef R_DEBUG_ALL
		show_current_layer("maxpool");
		printf("\tin: %d %d %d %d out: %d %d %d %d\n",in->dim[0],in->dim[1],in->dim[2],in->dim[3],out->dim[0],out->dim[1],out->dim[2],out->dim[3]);
	#endif	
		return;
	} else {
	#ifdef R_DEBUG
		show_current_layer("maxpool");
		printf("\tin: %d %d %d %d out: %d %d %d %d\n",in->dim[0],in->dim[1],in->dim[2],in->dim[3],out->dim[0],out->dim[1],out->dim[2],out->dim[3]);
	#endif	
	}
#ifndef NOT_RUN
	if(csi_maxpool_init(in, out, param) == CSINN_TRUE)
		csi_maxpool(in, out, param);
#endif
}

void averagepool(struct csi_tensor* in, struct csi_tensor* out, struct pool_params *param){
	if(run_status == 0){
	#ifdef R_DEBUG_ALL
		show_current_layer("avgpool");
		printf("\tin: %d %d %d %d out: %d %d %d %d\n",in->dim[0],in->dim[1],in->dim[2],in->dim[3],out->dim[0],out->dim[1],out->dim[2],out->dim[3]);
	#endif	
		return;
	} else {
	#ifdef R_DEBUG
		show_current_layer("avgpool");
		printf("\tin: %d %d %d %d out: %d %d %d %d\n",in->dim[0],in->dim[1],in->dim[2],in->dim[3],out->dim[0],out->dim[1],out->dim[2],out->dim[3]);
	#endif	
	}
#ifndef NOT_RUN
	if(csi_averagepool_init(in, out, param) == CSINN_TRUE)
		csi_averagepool(in, out, param);
#endif
}

void fullyconnected(struct csi_tensor* in, struct csi_tensor* out, struct csi_tensor* weight, struct csi_tensor* bias, struct fc_params *param){
	if(run_status == 0){
	#ifdef R_DEBUG_ALL
		show_current_layer("fc");
	#endif
		return;
	} else {
	#ifdef R_DEBUG
		show_current_layer("fc");
		printf("\tin: %d %d %d %d out: %d %d %d %d\n",in->dim[0],in->dim[1],in->dim[2],in->dim[3],out->dim[0],out->dim[1],out->dim[2],out->dim[3]);
		printf("\tweight: %d %d %d %d bias: %d %d %d %d\n",weight->dim[0],weight->dim[1],weight->dim[2],weight->dim[3],bias->dim[0],bias->dim[1],bias->dim[2],bias->dim[3]);
	#endif
	}
#ifndef NOT_RUN
	if(csi_fullyconnected_init(in, out, weight, bias, param) == CSINN_TRUE)
		csi_fullyconnected(in, out, weight, bias, param);
#endif
}

void basic_block(int planes, int stride, int downsample){
	struct csi_tensor *identity = CURRENT;
	conv2d(IN, OUT, KERNEL, BIAS, conv_param(stride, 1));
	relu(IN, OUT, relu_param());
	
	conv2d(IN, OUT, KERNEL, BIAS, conv_param(1, 1));
	struct csi_tensor *branch = CURRENT;

	if(downsample == TRUE){
		conv2d(identity, OUT, KERNEL, BIAS, conv_param(stride, 0));
		identity = CURRENT;
	}

	add(branch, identity, OUT, add_param());
	relu(IN, OUT, relu_param());
}

void make_layer(int planes, int stride){
	int downsample = FALSE;
	if(stride != 1 || CURRENT->dim[3] != planes){  //NCHW  c!=stride
		downsample = TRUE;
	}
	basic_block(planes, stride, downsample);
	basic_block(planes, 1, FALSE);
}

void resnet18(){
	conv2d(IN, OUT, KERNEL, BIAS, conv_param(2, 3));
	relu(IN, OUT, relu_param());
	maxpool(IN, OUT, pool_param(3,2,1));  //struct pool_params *params

	make_layer(64,1);
	make_layer(128,2);
	make_layer(256,2);
	make_layer(512,2);

	averagepool(IN, OUT, pool_param(7,1,0));
	fullyconnected(IN, OUT, WEIGHT, BIAS, fc_param());
}
void make_blank_mid_results_array(){
	add_mid_result_nchw(1, 3, 224, 224, "input");
	add_mid_result_nchw(1, 64, 122, 122, "conv1");
	add_mid_result_nchw(1, 64, 122, 122, "relu");
	add_mid_result_nchw(1, 64, 56, 56, "maxpool");
	add_mid_result_nchw(1, 64, 56, 56, "layer1.0.conv1");
	add_mid_result_nchw(1, 64, 56, 56, "layer1.0.relu");
	add_mid_result_nchw(1, 64, 56, 56, "layer1.0.conv2");
//	add_mid_result_nchw(1, 64, 56, 56, "layer1.downsample.0");
	add_mid_result_nchw(1, 64, 56, 56, "layer1.0.add");
	add_mid_result_nchw(1, 64, 56, 56, "layer1.0.relu");
	add_mid_result_nchw(1, 64, 56, 56, "layer1.1.conv1");
	add_mid_result_nchw(1, 64, 56, 56, "layer1.1.relu");
	add_mid_result_nchw(1, 64, 56, 56, "layer1.1.conv2");
	add_mid_result_nchw(1, 64, 56, 56, "layer1.1.add");
	add_mid_result_nchw(1, 64, 56, 56, "layer1.1.relu");
	add_mid_result_nchw(1, 128, 28, 28, "layer2.0.conv1");
	add_mid_result_nchw(1, 128, 28, 28, "layer2.0.relu");
	add_mid_result_nchw(1, 128, 28, 28, "layer2.0.conv2");
	add_mid_result_nchw(1, 128, 28, 28, "layer2.0.downsample.0");
	add_mid_result_nchw(1, 128, 28, 28, "layer2.0.add");
	add_mid_result_nchw(1, 128, 28, 28, "layer2.0.relu");
	add_mid_result_nchw(1, 128, 28, 28, "layer2.1.conv1");
	add_mid_result_nchw(1, 128, 28, 28, "layer2.1.relu");
	add_mid_result_nchw(1, 128, 28, 28, "layer2.1.conv2");
	add_mid_result_nchw(1, 128, 28, 28, "layer2.1.add");
	add_mid_result_nchw(1, 128, 28, 28, "layer2.1.relu");
	add_mid_result_nchw(1, 256, 14, 14, "layer3.0.conv1");
	add_mid_result_nchw(1, 256, 14, 14, "layer3.0.relu");
	add_mid_result_nchw(1, 256, 14, 14, "layer3.0.conv2");
	add_mid_result_nchw(1, 256, 14, 14, "layer3.0.downsample.0");
	add_mid_result_nchw(1, 256, 14, 14, "layer3.0.add");
	add_mid_result_nchw(1, 256, 14, 14, "layer3.0.relu");
	add_mid_result_nchw(1, 256, 14, 14, "layer3.1.conv1");
	add_mid_result_nchw(1, 256, 14, 14, "layer3.1.relu");
	add_mid_result_nchw(1, 256, 14, 14, "layer3.1.conv2");
	add_mid_result_nchw(1, 256, 14, 14, "layer3.1.add");
	add_mid_result_nchw(1, 256, 14, 14, "layer3.1.relu");
	add_mid_result_nchw(1, 512, 7, 7, "layer4.0.conv1");
	add_mid_result_nchw(1, 512, 7, 7, "layer4.0.relu");
	add_mid_result_nchw(1, 512, 7, 7, "layer4.0.conv2");
	add_mid_result_nchw(1, 512, 7, 7, "layer4.0.downsample.0");
	add_mid_result_nchw(1, 512, 7, 7, "layer4.0.add");
	add_mid_result_nchw(1, 512, 7, 7, "layer4.0.relu");
	add_mid_result_nchw(1, 512, 7, 7, "layer4.1.conv1");
	add_mid_result_nchw(1, 512, 7, 7, "layer4.1.relu");
	add_mid_result_nchw(1, 512, 7, 7, "layer4.1.conv2");
	add_mid_result_nchw(1, 512, 7, 7, "layer4.1.add");
	add_mid_result_nchw(1, 512, 7, 7, "layer4.1.relu");
	add_mid_result_nchw(1, 512, 1, 1, "avgpool");
	add_mid_result_nchw(1, 1, 1, 1000, "fc");   //1*512 x 512*1000 + bias, weight 1000*512
}

void make_mid_need_params(){
	add_compute_params_nchw(64, 64, 7, 7,     "conv1.kernel");
	add_compute_params_nchw(1, 64, 122, 122, "conv1.bias");
	add_compute_params_nchw(64, 64, 3, 3,     "layer1.0.conv1.kernel");
	add_compute_params_nchw(1, 64, 56, 56,   "layer1.0.conv1.bias");
	add_compute_params_nchw(64, 64, 3, 3, 	 "layer1.0.conv2.kernel");
	add_compute_params_nchw(1, 64, 56, 56,   "layer1.0.conv2.bias");
//	add_compute_params_nchw(64, 64, 3, 3, 	 "layer1.downsample.0.kernel");
//	add_compute_params_nchw(1, 64, 56, 56,   "layer1.downsample.0.bias");
	add_compute_params_nchw(64, 64, 3, 3, 	 "layer1.1.conv1.kernel");
	add_compute_params_nchw(1, 64, 56, 56, 	 "layer1.1.conv1.bias");
	add_compute_params_nchw(128, 64, 3, 3, 	 "layer1.1.conv2.kernel");
	add_compute_params_nchw(1, 64, 56, 56, 	 "layer1.1.conv2.bias");
	add_compute_params_nchw(128, 128, 3, 3,    "layer2.0.conv1.kernel");
	add_compute_params_nchw(1, 128, 28, 28,  "layer2.0.conv1.bias");
	add_compute_params_nchw(128, 128, 3, 3,    "layer2.0.conv2.kernel");
	add_compute_params_nchw(1, 128, 28, 28,  "layer2.0.conv2.bias");
	add_compute_params_nchw(128, 128, 1, 1,    "layer2.0.downsample.conv1.kernel");
	add_compute_params_nchw(1, 128, 28, 28,  "layer2.0.downsample.conv1.bias");
	add_compute_params_nchw(128, 128, 3, 3,    "layer2.1.conv1.kernel");
	add_compute_params_nchw(1, 128, 28, 28,  "layer2.1.conv1.bias");
	add_compute_params_nchw(256, 128, 3, 3,    "layer2.1.conv2.kernel");
	add_compute_params_nchw(1, 128, 28, 28,  "layer2.1.conv2.bias");
	add_compute_params_nchw(256, 256, 3, 3,    "layer3.0.conv1.kernel");
	add_compute_params_nchw(1, 256, 14, 14,  "layer3.0.conv1.bias");
	add_compute_params_nchw(256, 256, 3, 3,    "layer3.0.conv2.kernel");
	add_compute_params_nchw(1, 256, 14, 14,  "layer3.0.conv2.bias");
	add_compute_params_nchw(256, 256, 1, 1,    "layer3.0.downsample.conv1.kernel");
	add_compute_params_nchw(1, 256, 14, 14,  "layer3.0.downsample.conv1.bias");
	add_compute_params_nchw(256, 256, 3, 3,    "layer3.1.conv1.kernel");
	add_compute_params_nchw(1, 256, 14, 14,  "layer3.1.conv1.bias");
	add_compute_params_nchw(512, 256, 3, 3,    "layer3.1.conv2.kernel");
	add_compute_params_nchw(1, 256, 14, 14,  "layer3.1.conv2.bias");
	add_compute_params_nchw(512, 512, 3, 3,    "layer4.0.conv1.kernel");
	add_compute_params_nchw(1, 512, 7, 7,	 "layer4.0.conv1.bias");
	add_compute_params_nchw(512, 512, 3, 3,    "layer4.0.conv2.kernel");
	add_compute_params_nchw(1, 512, 7, 7,	 "layer4.0.conv2.bias");
	add_compute_params_nchw(512, 512, 1, 1,    "layer4.0.downsample.conv1.kernel");
	add_compute_params_nchw(1, 512, 7, 7,	 "layer4.0.downsample.conv1.bias");
	add_compute_params_nchw(512, 512, 3, 3,    "layer4.1.conv1.kernel");
	add_compute_params_nchw(1, 512, 7, 7,	 "layer4.1.conv1.bias");
	add_compute_params_nchw(512, 512, 3, 3,    "layer4.1.conv2.kernel");
	add_compute_params_nchw(1, 512, 7, 7,	 "layer4.1.conv2.bias");
	add_compute_params_nchw(1, 1, 1000, 512, "fc.weight");
	add_compute_params_nchw_i32(1, 1, 1, 1000,	 "fc.bias");

}

void init(){
	printf("[info] pre-run mid-space started\n");
	make_blank_mid_results_array();  //中间结果空数组空间预分配(包括input)
	make_mid_need_params();	//所需kernel和bias
//	add_mid_result_nchw(1, 3, 224, 224, "input");
//	printf("all layers: %d\n\n", all_mid_results_count);
}
void destroy(){
	for(int i=0;i<all_mid_results_count;i++){
		free(mid_results_array[i]->data);
		csi_free_tensor(mid_results_array[i]);
	}
	for(int i=0;i<all_mid_need_params_count;i++){
		free(mid_need_params[i]->data);
		csi_free_tensor(mid_need_params[i]);
	}
	for(int i=0;i<all_mid_params_config_count;i++)
		free(mid_params_config[i]);
	all_mid_results_count = 0;
	all_mid_need_params_count = 0;
	all_mid_params_config_count = 0;
	mid_results_count = 0;
	mid_need_params_count = 0;
	mid_params_config_count = 0;
}
int main(){
	init();
#if 1
	printf("[info] pre-run conf started\n");
	resnet18();  //pre-run to alloc the configs
	run_status = 1;	//now will run
	reset_all_counters();
	printf("[info] infer started\n");

	struct timeval new1,new2;
	gettimeofday (&new1, NULL);
	resnet18();
	gettimeofday (&new2, NULL);
	printf("ResNet 18 i8, time used: %f second \n\n",(new2.tv_sec-new1.tv_sec)+(new2.tv_usec-new1.tv_usec)/1000000.0);
#else
	conv2d(mid_results_array[3], mid_results_array[3],mid_need_params[2],mid_need_params[3], conv_param(64, 1));
#endif	
	//use CURRENT to get the result;
	printf("[info] destroy model\n");
	destroy();
	printf("[info] all succeed\n");
}
