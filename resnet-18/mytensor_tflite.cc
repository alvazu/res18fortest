#include <sys/time.h>
#include "operators.cc"

#define FALSE 0
#define TRUE 1
#define IN mid_results_array[mid_results_count]
#define CURRENT mid_results_array[mid_results_count]
#define OUT mid_results_array[mid_results_count+1]
#define FCBIAS mid_need_params[101]
#define FCWEIGHT mid_need_params[100]
#define MULTI mid_need_params[mid_need_params_count+0]
#define SHIFT mid_need_params[mid_need_params_count+1]
#define FILTER mid_need_params[mid_need_params_count+2]
#define IM2COL mid_need_params[mid_need_params_count+4]
#define BIAS mid_need_params[mid_need_params_count+3]

#define INIT_IN input()
#define INIT_OUT mid_results_array[mid_results_count+1]

#define MAX_LENGTH 122

//#define R_DEBUG
//#define R_DEBUG_ALL
//#define NOT_RUN
#define PRINT100 printf("\tmid_need_params[100]=%d(%d, %d), mid_need_params[101]=%d(%d)\n",mid_need_params[100]->shape->DimensionsCount(),mid_need_params[100]->shape->Dims(0),mid_need_params[100]->shape->Dims(1),mid_need_params[101]->shape->DimensionsCount(),mid_need_params[101]->shape->Dims(0))


struct TensorX{
	void* pdata;
	RuntimeShape* shape;
};

int run_status = 0;

TensorX *mid_results_array[MAX_LENGTH];
TensorX *mid_need_params[MAX_LENGTH];
void *mid_params_config[MAX_LENGTH];

char mid_results_str[MAX_LENGTH][255] = { 0 };


int mid_results_count = 0;	//current pointer of stack
int mid_need_params_count = 0;
int mid_params_config_count = 0;

int all_mid_results_count = 0; //length of stack
int all_mid_need_params_count = 0;
int all_mid_params_config_count = 0;

CpuBackendContext cpuBackendContext;



void add_mid_result_nchw(int n,int c,int h,int w,const char*name){
	const int outsh[4]={n,h,w,c};
	TensorX* tensor = (TensorX*)malloc(sizeof(TensorX));
	tensor->pdata = malloc(n*c*h*w*sizeof(int8));
	tensor->shape = new RuntimeShape(4,outsh);
	
	mid_results_array[all_mid_results_count++] = tensor;
	strcpy(mid_results_str[all_mid_results_count - 1],name);
}


void add_compute_params_conv(int N, int C, int CORE, int W, int W2, int F, int P, int S,const char*name){
//1,64,64,224,224,7,3,2,"conv1"
	const int OW = (W-F+2*P)/S+1;
	const int filtersh[4]={CORE,F,F,C};
	const int im2colsh[4]={N,OW,OW,F*F*C};
	RuntimeShape *convShapeBias = new RuntimeShape(1,CORE);
	RuntimeShape *convShapeFilter = new RuntimeShape(4,filtersh);
	RuntimeShape *convShapeIm2Col = new RuntimeShape(4,im2colsh);

	bool useIm2Col = S!=1 || F!=1;

	TensorX* out_multi = (TensorX*)malloc(sizeof(TensorX));
	TensorX* out_shift = (TensorX*)malloc(sizeof(TensorX));
	TensorX* bias = (TensorX*)malloc(sizeof(TensorX));
	TensorX* filter = (TensorX*)malloc(sizeof(TensorX));
	TensorX* im2col = (TensorX*)malloc(sizeof(TensorX));

	out_multi->pdata = malloc(N*N*CORE*sizeof(int));
	out_shift->pdata = malloc(N*N*CORE*sizeof(int));
	bias->pdata = malloc(N*N*CORE*sizeof(int32));
	filter->pdata = malloc(F*F*C*CORE*sizeof(int8));
	im2col->pdata = useIm2Col? malloc(N*W*W*F*F*C/S/S*sizeof(int8)): nullptr;
//printf("all_mid_need_params_count = %d ,useIm2Col %d, pdata = %p\n",all_mid_need_params_count,useIm2Col,im2col->pdata);

	out_multi->shape = nullptr;
	out_shift->shape = nullptr;
	bias->shape = convShapeBias;
	filter->shape = convShapeFilter;
	im2col->shape = convShapeIm2Col;

	mid_need_params[all_mid_need_params_count++] = out_multi;
	mid_need_params[all_mid_need_params_count++] = out_shift;
	mid_need_params[all_mid_need_params_count++] = filter;
	mid_need_params[all_mid_need_params_count++] = bias;
	mid_need_params[all_mid_need_params_count++] = im2col;
//printf("im2col %p\n",(int8*)(mid_need_params[4]->pdata));
}

void add_compute_params_fc(int N, int K, int M,const char*name){   // N1, K1000, M512    //1*512 x 512*1000 + bias, weight 1000*512
	TensorX* weight = (TensorX*)malloc(sizeof(TensorX));
	TensorX* bias = (TensorX*)malloc(sizeof(TensorX));
	
	int sp[2]={K,M};
	weight->pdata = malloc(M*K*sizeof(int8));
	weight->shape = new RuntimeShape(2,sp);
	
	bias->pdata = malloc(K*sizeof(int32));
	bias->shape = new RuntimeShape(1,K);
	
	mid_need_params[all_mid_need_params_count++] = weight;
//printf("%d making fc weight dimcount=%d\n",all_mid_need_params_count-1,mid_need_params[all_mid_need_params_count-1]->shape->DimensionsCount());
	mid_need_params[all_mid_need_params_count++] = bias;
//printf("%d making fc bias dimcount=%d\n",all_mid_need_params_count-1,mid_need_params[all_mid_need_params_count-1]->shape->DimensionsCount());
//printf("\tmaking fc params, weight: %d %d bias: %d\n",weight->shape->Dims(0),weight->shape->Dims(1),bias->shape->Dims(0));
	
}

void reset_all_counters(){
	mid_results_count = 0;
	mid_need_params_count = 0;
	mid_params_config_count = 0;
}

ConvParams *conv_param(int S, int P){
	if (run_status == 0){
		ConvParams *convParams = (ConvParams*)calloc(1, sizeof(ConvParams));
		//conv_params[conv_params_count++] = param;

		convParams->stride_width = S;
		convParams->stride_height = S;
		convParams->quantized_activation_min = -128;
		convParams->quantized_activation_max = 127;
		convParams->dilation_width_factor = 1;
		convParams->dilation_height_factor =1;
		enum PaddingType padding_type;
		padding_type = PaddingType::kSame;
		convParams->padding_type= padding_type;//PaddingType.kSame;
		struct PaddingValues padding_values;
		padding_values.width = P;
		padding_values.height = P;
		convParams->padding_values = padding_values;
		mid_params_config[all_mid_params_config_count++] = (void *)convParams;
		return convParams;
	} else {
		return (ConvParams*)mid_params_config[mid_params_config_count++];
	}
}

ArithmeticParams *add_param(){
	if (run_status == 0){
		ArithmeticParams *param = (ArithmeticParams *)calloc(1, sizeof(ArithmeticParams));
		param->quantized_activation_min = 0;
		param->quantized_activation_max = 127;
		param->input1_shift= -1;
		param->input2_shift= -1;
		param->output_shift= -1;
		mid_params_config[all_mid_params_config_count++] = (void *)param;
		return param;
	} else {
		return (ArithmeticParams *)mid_params_config[mid_params_config_count++];
	}
}

PoolParams *pool_param(int size, int stride, int pad){
	if (run_status == 0){
		
		PoolParams *poolparams = (PoolParams *)calloc(1, sizeof(PoolParams));
		poolparams->stride_height = stride;
		poolparams->stride_width = stride;
		poolparams->filter_height = size;
		poolparams->filter_width = size;
		PaddingValues pads;
		pads.width = pad;
		pads.height= pad;
		poolparams->padding_values = pads;
		poolparams->quantized_activation_min = 0;
		poolparams->quantized_activation_max = 127;
		
		mid_params_config[all_mid_params_config_count++] = (void *)poolparams;
		return poolparams;
	} else {
		return (PoolParams*)mid_params_config[mid_params_config_count++];
	}
}


FullyConnectedParams *fc_param(){
	if (run_status == 0){
		FullyConnectedParams *fcParams = (FullyConnectedParams *)calloc(1, sizeof(FullyConnectedParams));
		fcParams->quantized_activation_min = -128;
		fcParams->quantized_activation_max = 127;
		fcParams->output_shift = 1;
		fcParams->output_multiplier = 1;
		
		mid_params_config[all_mid_params_config_count++] = (void *)fcParams;
		return fcParams;
	} else {
		return (FullyConnectedParams *)mid_params_config[mid_params_config_count++];
	}
}

void show_current_layer(const char* layer){
	printf("%d/%d, %d/%d ------------------------------------ [%s] [%s]\n", mid_results_count, all_mid_results_count, mid_need_params_count, all_mid_need_params_count, layer, mid_results_str[mid_results_count]);
}

void conv2d(TensorX *pinput, TensorX *poutput, TensorX *multi, TensorX *shift, TensorX *filter, TensorX *bias, TensorX *im2col, ConvParams *convParams){
	mid_results_count++;
	mid_need_params_count+=5;
	if(run_status == 0){
	#ifdef R_DEBUG_ALL
		show_current_layer("conv");
		printf("\tin: %d %d %d %d out: %d %d %d %d\n",pinput->shape->Dims(0),pinput->shape->Dims(1),pinput->shape->Dims(2),pinput->shape->Dims(3),poutput->shape->Dims(0),poutput->shape->Dims(1),poutput->shape->Dims(2),poutput->shape->Dims(3));
		printf("\tfilter: %d %d %d %d bias: %d\n",filter->shape->Dims(0),filter->shape->Dims(1),filter->shape->Dims(2),filter->shape->Dims(3),bias->shape->Dims(0));
		printf("\tim2col: %d %d %d %d\n",im2col->shape->Dims(0),im2col->shape->Dims(1),im2col->shape->Dims(2),im2col->shape->Dims(3));
	#endif
		return;
	} else {
	#ifdef R_DEBUG
		show_current_layer("conv");
		printf("\tin: %d %d %d %d out: %d %d %d %d\n",pinput->shape->Dims(0),pinput->shape->Dims(1),pinput->shape->Dims(2),pinput->shape->Dims(3),poutput->shape->Dims(0),poutput->shape->Dims(1),poutput->shape->Dims(2),poutput->shape->Dims(3));
		printf("\tfilter: %d %d %d %d bias: %d\n",filter->shape->Dims(0),filter->shape->Dims(1),filter->shape->Dims(2),filter->shape->Dims(3),bias->shape->Dims(0));
		printf("\tim2col: %d %d %d %d\n",im2col->shape->Dims(0),im2col->shape->Dims(1),im2col->shape->Dims(2),im2col->shape->Dims(3));
		PRINT100;
	#endif

	}
#ifndef NOT_RUN
/*printf("mid_need_params_count %d\n",mid_need_params_count);
printf("multi %p\n",(int*)(multi->pdata));
printf("shift %p\n",(int*)(shift->pdata));
printf("filter %p\n",(int8*)(filter->pdata));
printf("bias %p\n",(int32*)(bias->pdata));
printf("im2col %p\n",(int8*)(im2col->pdata));*/
	ConvPerChannel(
		*convParams,
		(int*)(multi->pdata),
		(int*)(shift->pdata),
		*(pinput->shape),
		(int8*)(pinput->pdata),
		*(filter->shape),
		(int8*)(filter->pdata),
		*(bias->shape),
		(int32*)(bias->pdata),
		*(poutput->shape),
		(int8*)(poutput->pdata),
		*(im2col->shape),
		(int8*)(im2col->pdata),
		&cpuBackendContext
	);
#endif
}




void relu(TensorX* in, TensorX* out){
	mid_results_count++;
	if(run_status == 0){
	#ifdef R_DEBUG_ALL
		show_current_layer("relu");
		printf("\tin: %d %d %d %d out: %d %d %d %d\n",in->shape->Dims(0),in->shape->Dims(1),in->shape->Dims(2),in->shape->Dims(3),out->shape->Dims(0),out->shape->Dims(1),out->shape->Dims(2),out->shape->Dims(3));
	#endif	
		return;
	} else {
	#ifdef R_DEBUG
		show_current_layer("relu");
		printf("\tin: %d %d %d %d out: %d %d %d %d\n",in->shape->Dims(0),in->shape->Dims(1),in->shape->Dims(2),in->shape->Dims(3),out->shape->Dims(0),out->shape->Dims(1),out->shape->Dims(2),out->shape->Dims(3));
		PRINT100;
	#endif	
	}
#ifndef NOT_RUN
	Relu<int8>(*(in->shape),(int8*)(in->pdata),*(out->shape),(int8*)(out->pdata));
#endif
}

void add(TensorX* a, TensorX* b, TensorX* out, ArithmeticParams* param){
	mid_results_count++;
	if(run_status == 0){
	#ifdef R_DEBUG_ALL
		show_current_layer("add");
		printf("\ta: %d %d %d %d, b: %d %d %d %d, out: %d %d %d %d\n",a->shape->Dims(0),a->shape->Dims(1),a->shape->Dims(2),a->shape->Dims(3),b->shape->Dims(0),b->shape->Dims(1),b->shape->Dims(2),b->shape->Dims(3),out->shape->Dims(0),out->shape->Dims(1),out->shape->Dims(2),out->shape->Dims(3));
	#endif	
		return;
	} else {
	#ifdef R_DEBUG
		show_current_layer("add");
		printf("\ta: %d %d %d %d, b: %d %d %d %d, out: %d %d %d %d\n",a->shape->Dims(0),a->shape->Dims(1),a->shape->Dims(2),a->shape->Dims(3),b->shape->Dims(0),b->shape->Dims(1),b->shape->Dims(2),b->shape->Dims(3),out->shape->Dims(0),out->shape->Dims(1),out->shape->Dims(2),out->shape->Dims(3));
		PRINT100;
	#endif	
	}
#ifndef NOT_RUN
	Add(*param,
		*(a->shape),
		(int8*)(a->pdata),
		*(b->shape),
		(int8*)(b->pdata),
		*(out->shape),
		(int8*)(out->pdata)
	);
#endif
}

void maxpool(TensorX* in, TensorX* out, PoolParams *param){
	mid_results_count++;
	if(run_status == 0){
	#ifdef R_DEBUG_ALL
		show_current_layer("maxpool");
		printf("\tin: %d %d %d %d out: %d %d %d %d\n",in->shape->Dims(0),in->shape->Dims(1),in->shape->Dims(2),in->shape->Dims(3),out->shape->Dims(0),out->shape->Dims(1),out->shape->Dims(2),out->shape->Dims(3));
	#endif	
		return;
	} else {
	#ifdef R_DEBUG
		show_current_layer("maxpool");
		printf("\tin: %d %d %d %d out: %d %d %d %d\n",in->shape->Dims(0),in->shape->Dims(1),in->shape->Dims(2),in->shape->Dims(3),out->shape->Dims(0),out->shape->Dims(1),out->shape->Dims(2),out->shape->Dims(3));
		PRINT100;
	#endif	
	}
#ifndef NOT_RUN
	MaxPool(*param,*(in->shape),(int8*)(in->pdata),*(out->shape),(int8*)(out->pdata));
#endif
}

void averagepool(TensorX* in, TensorX* out, PoolParams *param){
	mid_results_count++;
	if(run_status == 0){
	#ifdef R_DEBUG_ALL
		show_current_layer("avgpool");
		printf("\tin: %d %d %d %d out: %d %d %d %d\n",in->shape->Dims(0),in->shape->Dims(1),in->shape->Dims(2),in->shape->Dims(3),out->shape->Dims(0),out->shape->Dims(1),out->shape->Dims(2),out->shape->Dims(3));
	#endif	
		return;
	} else {
	#ifdef R_DEBUG
		show_current_layer("avgpool");
		printf("\tin: %d %d %d %d out: %d %d %d %d\n",in->shape->Dims(0),in->shape->Dims(1),in->shape->Dims(2),in->shape->Dims(3),out->shape->Dims(0),out->shape->Dims(1),out->shape->Dims(2),out->shape->Dims(3));
		PRINT100;
	#endif	
	}
#ifndef NOT_RUN
	AveragePool(*param,*(in->shape),(int8*)(in->pdata),*(out->shape),(int8*)(out->pdata));
#endif
}

void fullyconnected(TensorX* in, TensorX* out, TensorX* weight, TensorX* bias, FullyConnectedParams *param){
	mid_results_count++;
	if(run_status == 0){
	#ifdef R_DEBUG_ALL
		show_current_layer("fc");
		printf("\tin: %d %d %d %d out: %d %d %d %d\n",in->shape->Dims(0),in->shape->Dims(1),in->shape->Dims(2),in->shape->Dims(3),out->shape->Dims(0),out->shape->Dims(1),out->shape->Dims(2),out->shape->Dims(3));
		printf("\t%d fc weight dimcount=%d\n",mid_need_params_count,weight->shape->DimensionsCount());
		printf("\t%d fc bias dimcount=%d\n",mid_need_params_count,bias->shape->DimensionsCount());
	#endif
		mid_need_params_count+=2;
		return;
	} else {
	#ifdef R_DEBUG
		show_current_layer("fc");
		printf("\tin: %d %d %d %d out: %d %d %d %d\n",in->shape->Dims(0),in->shape->Dims(1),in->shape->Dims(2),in->shape->Dims(3),out->shape->Dims(0),out->shape->Dims(1),out->shape->Dims(2),out->shape->Dims(3));
		printf("\tweight(%d): %d %d bias(%d): %d\n",weight->shape->DimensionsCount(),weight->shape->Dims(0),weight->shape->Dims(1),bias->shape->DimensionsCount(),bias->shape->Dims(0));
		printf("\t%d fc weight dimcount=%d\n",mid_need_params_count,weight->shape->DimensionsCount());
		printf("\t%d fc bias dimcount=%d\n",mid_need_params_count,bias->shape->DimensionsCount());
		PRINT100;
	#endif
	}
	mid_need_params_count+=2;
#ifndef NOT_RUN

	FullyConnected(
		*param, 
		*(in->shape),
		(int8*)(in->pdata), 
		*(weight->shape), 
		(int8*)(weight->pdata), 
		*(bias->shape), 
		(int32*)(bias->pdata), 
		*(out->shape), 
		(int8*)(out->pdata), 
		&cpuBackendContext
	);
#endif
}

void basic_block(int planes, int stride, int downsample){
	TensorX *identity = CURRENT;
	conv2d(IN, OUT, MULTI, SHIFT, FILTER, BIAS, IM2COL, conv_param(stride, 1));
	relu(IN, OUT);

	conv2d(IN, OUT, MULTI, SHIFT, FILTER, BIAS, IM2COL, conv_param(1, 1));
	TensorX *branch = CURRENT;

	if(downsample == TRUE){
		conv2d(identity, OUT, MULTI, SHIFT, FILTER, BIAS, IM2COL, conv_param(stride, 0));
		identity = CURRENT;
	}

	add(branch, identity, OUT, add_param());
	relu(IN, OUT);
}

void make_layer(int planes, int stride){
	int downsample = FALSE;
	if(stride != 1 || CURRENT->shape->Dims(3) != planes){  //NCHW  c!=stride
		downsample = TRUE;
	}
	basic_block(planes, stride, downsample);
	basic_block(planes, 1, FALSE);

}

void resnet18(){
	conv2d(IN, OUT, MULTI, SHIFT, FILTER, BIAS, IM2COL, conv_param(2, 3));
	relu(IN, OUT);
	maxpool(IN, OUT, pool_param(3,2,1));  //struct pool_params *params

	make_layer(64,1);
	make_layer(128,2);
	make_layer(256,2);
	make_layer(512,2);

	averagepool(IN, OUT, pool_param(7,1,0));
#if 1
	fullyconnected(IN, OUT, FCWEIGHT, FCBIAS, fc_param());
#endif
}
void make_blank_mid_results_array(){
	add_mid_result_nchw(1, 3, 224, 224, "input");
	add_mid_result_nchw(1, 64, 112, 112, "conv1");
	add_mid_result_nchw(1, 64, 112, 112, "relu");
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
	add_mid_result_nchw(1, 1000, 1, 1, "fc");   //1*512 x 512*1000 + bias, weight 1000*512
}

void make_mid_need_params(){
	add_compute_params_conv(1,64,64,224,224,7,3,2,"conv1");
	add_compute_params_conv(1,64,64,56,56,3,1,1,"layer1.0.conv1");
	add_compute_params_conv(1,64,64,56,56,3,1,1,"layer1.0.conv2");
	add_compute_params_conv(1,64,64,56,56,3,1,1,"layer1.1.conv1");
	add_compute_params_conv(1,64,64,56,56,3,1,1,"layer1.1.conv2");

	add_compute_params_conv(1,64,128,56,56,3,1,2,"layer2.0.conv1");
	add_compute_params_conv(1,128,128,28,28,3,1,1,"layer2.0.conv2");
	add_compute_params_conv(1,64,128,56,56,1,0,2,"layer2.0.downsample.0");
	add_compute_params_conv(1,128,128,28,28,3,1,1,    "layer2.1.conv1");
	add_compute_params_conv(1,128,128,28,28,3,1,1,    "layer2.1.conv2");

	add_compute_params_conv(1,128,256,28,28,3,1,2,  "layer3.0.conv1");
	add_compute_params_conv(1,256,256,14,14,3,1,1,  "layer3.0.conv2");
	add_compute_params_conv(1,128,256,28,28,1,0,2,    "layer3.0.downsample.conv1");
	add_compute_params_conv(1,256,256,14,14,3,1,1,    "layer3.1.conv1");
	add_compute_params_conv(1,256,256,14,14,3,1,1,    "layer3.1.conv2");

	add_compute_params_conv(1,256,512,14,14,3,1,2,    "layer4.0.conv1");
	add_compute_params_conv(1,512,512,7,7,3,1,1,    "layer4.0.conv2");
	add_compute_params_conv(1,256,512,14,14,1,0,2,    "layer4.0.downsample.conv1");
	add_compute_params_conv(1,512,512,7,7,3,1,1,    "layer4.1.conv1");
	add_compute_params_conv(1,512,512,7,7,3,1,1,    "layer4.1.conv2");
	add_compute_params_fc(1, 1000, 512,	"fc");
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
		if(mid_results_array[i]->pdata != nullptr)
			free(mid_results_array[i]->pdata);
		if(mid_results_array[i]->shape != nullptr)
			delete mid_results_array[i]->shape;
		if(mid_results_array[i] != nullptr)
			free(mid_results_array[i]);
	}
	for(int i=0;i<all_mid_need_params_count;i++){
		if(mid_need_params[i]->pdata != nullptr)
			free(mid_need_params[i]->pdata);
		if(mid_need_params[i]->shape != nullptr)
			delete mid_need_params[i]->shape;
		if(mid_need_params[i] != nullptr)
			free(mid_need_params[i]);
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
	cpuBackendContext.SetUseCaching(false);
	printf("[info] init started\n");
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

