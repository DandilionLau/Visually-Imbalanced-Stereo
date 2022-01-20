//Implement the backforward function by Jianbo Liu and Yicun Liu
//Update the whole implemention to support arbitrary dimension filter
//    size by Jianbo Liu and Yicun Liu
#include <THC.h>

#include <THCGeneral.h>
#include "stdio.h"

#define VEC_0(ARRAY) ((ARRAY).x)
#define VEC_1(ARRAY) ((ARRAY).y)
#define VEC_2(ARRAY) ((ARRAY).z)
#define VEC_3(ARRAY) ((ARRAY).w)

#define IDX_1(ARRAY, X)          ((ARRAY)[((X) * (ARRAY##_stride.x))])
#define IDX_2(ARRAY, X, Y)       ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y))])
#define IDX_3(ARRAY, X, Y, Z)    ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y)) + ((Z) * (ARRAY##_stride.z))])
#define IDX_4(ARRAY, X, Y, Z, W) ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y)) + ((Z) * (ARRAY##_stride.z)) + ((W) * (ARRAY##_stride.w))])

#ifdef __cplusplus
	extern "C" {
#endif

//Define forward operations
__global__ void kernel_SeparableConvolution_updateOutput_forward_general(
	const int n,
	const float* input, const long4 input_size, const long4 input_stride,
	const float* vertical, const long4 vertical_size, const long4 vertical_stride,
	const float* horizontal, const long4 horizontal_size, const long4 horizontal_stride,
	float* output, const long4 output_size, const long4 output_stride
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n) {
		return;
	}

    int filterSize_v = vertical_size.y;
    int filterSize_h = horizontal_size.y;

	float dblOutput = 0.0;

	int intBatch = ( intIndex / VEC_3(output_size) / VEC_2(output_size) / VEC_1(output_size) ) % VEC_0(output_size);
	int intDepth = ( intIndex / VEC_3(output_size) / VEC_2(output_size)                      ) % VEC_1(output_size);
	int intY     = ( intIndex / VEC_3(output_size)                                           ) % VEC_2(output_size);
	int intX     = ( intIndex                                                                ) % VEC_3(output_size);

	for (int intFilterY = 0; intFilterY < filterSize_v; intFilterY += 1) {
		for (int intFilterX = 0; intFilterX < filterSize_h; intFilterX += 1) {
			dblOutput += IDX_4(input, intBatch, intDepth, intY + intFilterY, intX + intFilterX) * IDX_4(vertical, intBatch, intFilterY, intY, intX) * IDX_4(horizontal, intBatch, intFilterX, intY, intX);
		}
	}

	output[intIndex] = dblOutput;
}

__global__ void kernel_SeparableConvolution_updateOutput_forward_only_horizontal(
	const int n,
	const float* input, const long4 input_size, const long4 input_stride,
	const float* horizontal, const long4 horizontal_size, const long4 horizontal_stride,
	float* output, const long4 output_size, const long4 output_stride
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n) {
		return;
	}

    int filterSize_h = horizontal_size.y;

	float dblOutput = 0.0;

	int intBatch = ( intIndex / VEC_3(output_size) / VEC_2(output_size) / VEC_1(output_size) ) % VEC_0(output_size);
	int intDepth = ( intIndex / VEC_3(output_size) / VEC_2(output_size)                      ) % VEC_1(output_size);
	int intY     = ( intIndex / VEC_3(output_size)                                           ) % VEC_2(output_size);
	int intX     = ( intIndex                                                                ) % VEC_3(output_size);

	for (int intFilterX = 0; intFilterX < filterSize_h; intFilterX += 1) {
			dblOutput += IDX_4(input, intBatch, intDepth, intY, intX + intFilterX) * IDX_4(horizontal, intBatch, intFilterX, intY, intX);
	}

	output[intIndex] = dblOutput;
}

__global__ void kernel_SeparableConvolution_updateOutput_forward_only_vertical(
	const int n,
	const float* input, const long4 input_size, const long4 input_stride,
	const float* vertical, const long4 vertical_size, const long4 vertical_stride,
	float* output, const long4 output_size, const long4 output_stride
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n) {
		return;
	}

    int filterSize_v = vertical_size.y;

	float dblOutput = 0.0;

	int intBatch = ( intIndex / VEC_3(output_size) / VEC_2(output_size) / VEC_1(output_size) ) % VEC_0(output_size);
	int intDepth = ( intIndex / VEC_3(output_size) / VEC_2(output_size)                      ) % VEC_1(output_size);
	int intY     = ( intIndex / VEC_3(output_size)                                           ) % VEC_2(output_size);
	int intX     = ( intIndex                                                                ) % VEC_3(output_size);

	for (int intFilterY = 0; intFilterY < filterSize_v; intFilterY += 1) {
		dblOutput += IDX_4(input, intBatch, intDepth, intY + intFilterY, intX) * IDX_4(vertical, intBatch, intFilterY, intY, intX);
	}

	output[intIndex] = dblOutput;
}

void SeparableConvolution_kernel_forward(
	THCState* state,
	THCudaTensor* input,
	THCudaTensor* vertical,
	THCudaTensor* horizontal,
	THCudaTensor* output
) {
	int n = 0;

	n = THCudaTensor_nElement(state, output);
	kernel_SeparableConvolution_updateOutput_forward_general<<< (n + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
		n,
		THCudaTensor_data(state, input), make_long4(input->size[0], input->size[1], input->size[2], input->size[3]), make_long4(input->stride[0], input->stride[1], input->stride[2], input->stride[3]),
		THCudaTensor_data(state, vertical), make_long4(vertical->size[0], vertical->size[1], vertical->size[2], vertical->size[3]), make_long4(vertical->stride[0], vertical->stride[1], vertical->stride[2], vertical->stride[3]),
		THCudaTensor_data(state, horizontal), make_long4(horizontal->size[0], horizontal->size[1], horizontal->size[2], horizontal->size[3]), make_long4(horizontal->stride[0], horizontal->stride[1], horizontal->stride[2], horizontal->stride[3]),
		THCudaTensor_data(state, output), make_long4(output->size[0], output->size[1], output->size[2], output->size[3]), make_long4(output->stride[0], output->stride[1], output->stride[2], output->stride[3])
	);

	THCudaCheck(cudaGetLastError());
}


void SeparableConvolution_kernel_forward_only_horizontal(
	THCState* state,
	THCudaTensor* input,
	THCudaTensor* horizontal,
	THCudaTensor* output
) {
	int n = 0;

	n = THCudaTensor_nElement(state, output);
	kernel_SeparableConvolution_updateOutput_forward_only_horizontal<<< (n + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
		n,
		THCudaTensor_data(state, input), make_long4(input->size[0], input->size[1], input->size[2], input->size[3]), make_long4(input->stride[0], input->stride[1], input->stride[2], input->stride[3]),
		THCudaTensor_data(state, horizontal), make_long4(horizontal->size[0], horizontal->size[1], horizontal->size[2], horizontal->size[3]), make_long4(horizontal->stride[0], horizontal->stride[1], horizontal->stride[2], horizontal->stride[3]),
		THCudaTensor_data(state, output), make_long4(output->size[0], output->size[1], output->size[2], output->size[3]), make_long4(output->stride[0], output->stride[1], output->stride[2], output->stride[3])
	);

	THCudaCheck(cudaGetLastError());
}


void SeparableConvolution_kernel_forward_only_vertical(
	THCState* state,
	THCudaTensor* input,
	THCudaTensor* vertical,
	THCudaTensor* output
) {
	int n = 0;

	n = THCudaTensor_nElement(state, output);
	kernel_SeparableConvolution_updateOutput_forward_only_vertical<<< (n + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
		n,
		THCudaTensor_data(state, input), make_long4(input->size[0], input->size[1], input->size[2], input->size[3]), make_long4(input->stride[0], input->stride[1], input->stride[2], input->stride[3]),
		THCudaTensor_data(state, vertical), make_long4(vertical->size[0], vertical->size[1], vertical->size[2], vertical->size[3]), make_long4(vertical->stride[0], vertical->stride[1], vertical->stride[2], vertical->stride[3]),
		THCudaTensor_data(state, output), make_long4(output->size[0], output->size[1], output->size[2], output->size[3]), make_long4(output->stride[0], output->stride[1], output->stride[2], output->stride[3])
	);

	THCudaCheck(cudaGetLastError());
}


//Define backward operations
__global__ void kernel_SeparableConvolution_updateOutput_backward_NN(
	const int n,
	const float* input, const long4 input_size, const long4 input_stride,
	const float* vertical, const long4 vertical_size, const long4 vertical_stride,
	const float* horizontal, const long4 horizontal_size, const long4 horizontal_stride,
	const float* grad_input, const long4 grad_input_size, const long4 grad_input_stride,
	float* output_h, float* output_v, const long4 output_size, const long4 output_stride
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n) {
		return;
	}

	float val_Output_h = 0.0;
	float val_Output_v = 0.0;

	int intBatch = ( intIndex / VEC_3(output_size) / VEC_2(output_size) / VEC_1(output_size) ) % VEC_0(output_size);
	int intDepth = ( intIndex / VEC_3(output_size) / VEC_2(output_size)                      ) % VEC_1(output_size);
	int intY     = ( intIndex / VEC_3(output_size)                                           ) % VEC_2(output_size);
	int intX     = ( intIndex                                                                ) % VEC_3(output_size);

    int filterSize_N = vertical_size.y;

	for (int intFilterN = 0; intFilterN < filterSize_N; intFilterN += 1) {
            float val_sum_channel = 0.0;
            val_sum_channel += IDX_4(input, intBatch, 0, intY + intDepth, intX + intFilterN) * IDX_4(grad_input, intBatch, 0, intY, intX);
            val_sum_channel += IDX_4(input, intBatch, 1, intY + intDepth, intX + intFilterN) * IDX_4(grad_input, intBatch, 1, intY, intX);
            val_sum_channel += IDX_4(input, intBatch, 2, intY + intDepth, intX + intFilterN) * IDX_4(grad_input, intBatch, 2, intY, intX);
			val_Output_v += val_sum_channel * IDX_4(horizontal, intBatch, intFilterN, intY, intX);
	}

	output_v[intIndex] = val_Output_v; // * 0.33333333f;

    
	for (int intFilterN = 0; intFilterN < filterSize_N; intFilterN += 1) {
            float val_sum_channel = 0.0;
            val_sum_channel += IDX_4(input, intBatch, 0, intY + intFilterN, intX + intDepth) * IDX_4(grad_input, intBatch, 0, intY, intX);
            val_sum_channel += IDX_4(input, intBatch, 1, intY + intFilterN, intX + intDepth) * IDX_4(grad_input, intBatch, 1, intY, intX);
            val_sum_channel += IDX_4(input, intBatch, 2, intY + intFilterN, intX + intDepth) * IDX_4(grad_input, intBatch, 2, intY, intX);
			val_Output_h += val_sum_channel * IDX_4(vertical, intBatch, intFilterN, intY, intX);
	}

	output_h[intIndex] = val_Output_h; // * 0.33333333f;
}

__global__ void kernel_SeparableConvolution_updateOutput_backward_horizontal0(
	const int n,
	const float* input, const long4 input_size, const long4 input_stride,
	const float* grad_input, const long4 grad_input_size, const long4 grad_input_stride,
	float* output_h, const long4 output_size, const long4 output_stride
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n) {
		return;
	}

	float val_Output_h = 0.0;

	int intBatch = ( intIndex / VEC_3(output_size) / VEC_2(output_size) / VEC_1(output_size) ) % VEC_0(output_size);
	int intDepth = ( intIndex / VEC_3(output_size) / VEC_2(output_size)                      ) % VEC_1(output_size);
	int intY     = ( intIndex / VEC_3(output_size)                                           ) % VEC_2(output_size);
	int intX     = ( intIndex                                                                ) % VEC_3(output_size);


    val_Output_h += IDX_4(input, intBatch, 0, intY, intX + intDepth) * IDX_4(grad_input, intBatch, 0, intY, intX);
    val_Output_h += IDX_4(input, intBatch, 1, intY, intX + intDepth) * IDX_4(grad_input, intBatch, 1, intY, intX);
    val_Output_h += IDX_4(input, intBatch, 2, intY, intX + intDepth) * IDX_4(grad_input, intBatch, 2, intY, intX);

	output_h[intIndex] = val_Output_h; // * 0.33333333f;
}


__global__ void kernel_SeparableConvolution_updateOutput_backward_horizontal1(
	const int n,
	const float* input, const long4 input_size, const long4 input_stride,
	const float* vertical, const long4 vertical_size, const long4 vertical_stride,
	const float* grad_input, const long4 grad_input_size, const long4 grad_input_stride,
	float* output_h, const long4 output_size, const long4 output_stride
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n) {
		return;
	}

	float val_Output_h = 0.0;

	int intBatch = ( intIndex / VEC_3(output_size) / VEC_2(output_size) / VEC_1(output_size) ) % VEC_0(output_size);
	int intDepth = ( intIndex / VEC_3(output_size) / VEC_2(output_size)                      ) % VEC_1(output_size);
	int intY     = ( intIndex / VEC_3(output_size)                                           ) % VEC_2(output_size);
	int intX     = ( intIndex                                                                ) % VEC_3(output_size);

    int filterSize_v = vertical_size.y;

	for (int intFilterN = 0; intFilterN < filterSize_v; intFilterN += 1) {
            float val_sum_channel = 0.0;
            val_sum_channel += IDX_4(input, intBatch, 0, intY + intFilterN, intX + intDepth) * IDX_4(grad_input, intBatch, 0, intY, intX);
            val_sum_channel += IDX_4(input, intBatch, 1, intY + intFilterN, intX + intDepth) * IDX_4(grad_input, intBatch, 1, intY, intX);
            val_sum_channel += IDX_4(input, intBatch, 2, intY + intFilterN, intX + intDepth) * IDX_4(grad_input, intBatch, 2, intY, intX);
			val_Output_h += val_sum_channel * IDX_4(vertical, intBatch, intFilterN, intY, intX);
	}

	output_h[intIndex] = val_Output_h; // * 0.33333333f;
}


__global__ void kernel_SeparableConvolution_updateOutput_backward_vertical0(
	const int n,
	const float* input, const long4 input_size, const long4 input_stride,
	const float* grad_input, const long4 grad_input_size, const long4 grad_input_stride,
	float* output_v, const long4 output_size, const long4 output_stride
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n) {
		return;
	}

	float val_Output_v = 0.0;

	int intBatch = ( intIndex / VEC_3(output_size) / VEC_2(output_size) / VEC_1(output_size) ) % VEC_0(output_size);
	int intDepth = ( intIndex / VEC_3(output_size) / VEC_2(output_size)                      ) % VEC_1(output_size);
	int intY     = ( intIndex / VEC_3(output_size)                                           ) % VEC_2(output_size);
	int intX     = ( intIndex                                                                ) % VEC_3(output_size);


    val_Output_v += IDX_4(input, intBatch, 0, intY + intDepth, intX) * IDX_4(grad_input, intBatch, 0, intY, intX);
    val_Output_v += IDX_4(input, intBatch, 1, intY + intDepth, intX) * IDX_4(grad_input, intBatch, 1, intY, intX);
    val_Output_v += IDX_4(input, intBatch, 2, intY + intDepth, intX) * IDX_4(grad_input, intBatch, 2, intY, intX);

	output_v[intIndex] = val_Output_v; // * 0.33333333f;
}


__global__ void kernel_SeparableConvolution_updateOutput_backward_vertical1(
	const int n,
	const float* input, const long4 input_size, const long4 input_stride,
	const float* horizontal, const long4 horizontal_size, const long4 horizontal_stride,
	const float* grad_input, const long4 grad_input_size, const long4 grad_input_stride,
	float* output_v, const long4 output_size, const long4 output_stride
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n) {
		return;
	}

	float val_Output_v = 0.0;

	int intBatch = ( intIndex / VEC_3(output_size) / VEC_2(output_size) / VEC_1(output_size) ) % VEC_0(output_size);
	int intDepth = ( intIndex / VEC_3(output_size) / VEC_2(output_size)                      ) % VEC_1(output_size);
	int intY     = ( intIndex / VEC_3(output_size)                                           ) % VEC_2(output_size);
	int intX     = ( intIndex                                                                ) % VEC_3(output_size);

    int filterSize_h = horizontal_size.y;

	for (int intFilterN = 0; intFilterN < filterSize_h; intFilterN += 1) {
            float val_sum_channel = 0.0;
            val_sum_channel += IDX_4(input, intBatch, 0, intY + intDepth, intX + intFilterN) * IDX_4(grad_input, intBatch, 0, intY, intX);
            val_sum_channel += IDX_4(input, intBatch, 1, intY + intDepth, intX + intFilterN) * IDX_4(grad_input, intBatch, 1, intY, intX);
            val_sum_channel += IDX_4(input, intBatch, 2, intY + intDepth, intX + intFilterN) * IDX_4(grad_input, intBatch, 2, intY, intX);
			val_Output_v += val_sum_channel * IDX_4(horizontal, intBatch, intFilterN, intY, intX);
	}

	output_v[intIndex] = val_Output_v; // * 0.33333333f;
}


void SeparableConvolution_kernel_backward(
	THCState* state,
	THCudaTensor* input,
	THCudaTensor* vertical,
	THCudaTensor* horizontal,
	THCudaTensor* grad_output,
	THCudaTensor* grad_vertical,
	THCudaTensor* grad_horizontal
) {
	int n_vertical = 0;
	n_vertical = THCudaTensor_nElement(state, grad_vertical);

	int n_horizontal = 0;
	n_horizontal = THCudaTensor_nElement(state, grad_horizontal);

    if(n_vertical == n_horizontal) {

	    kernel_SeparableConvolution_updateOutput_backward_NN<<< (n_vertical + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
		    n_vertical,
		    THCudaTensor_data(state, input), make_long4(input->size[0], input->size[1], input->size[2], input->size[3]), make_long4(input->stride[0], input->stride[1], input->stride[2], input->stride[3]),
		    THCudaTensor_data(state, vertical), make_long4(vertical->size[0], vertical->size[1], vertical->size[2], vertical->size[3]), make_long4(vertical->stride[0], vertical->stride[1], vertical->stride[2], vertical->stride[3]),
		    THCudaTensor_data(state, horizontal), make_long4(horizontal->size[0], horizontal->size[1], horizontal->size[2], horizontal->size[3]), make_long4(horizontal->stride[0], horizontal->stride[1], horizontal->stride[2], horizontal->stride[3]),
		    THCudaTensor_data(state, grad_output), make_long4(grad_output->size[0], grad_output->size[1], grad_output->size[2], grad_output->size[3]), make_long4(grad_output->stride[0], grad_output->stride[1], grad_output->stride[2], grad_output->stride[3]),
		    THCudaTensor_data(state, grad_horizontal), THCudaTensor_data(state, grad_vertical), make_long4(grad_horizontal->size[0], grad_horizontal->size[1], grad_horizontal->size[2], grad_horizontal->size[3]), make_long4(grad_horizontal->stride[0], grad_horizontal->stride[1], grad_horizontal->stride[2], grad_horizontal->stride[3])
        );
    
    } else {
        //when n_h != n_v: do vertical and horizontal gradient computation respectively
        
        //vertical gradient
	    kernel_SeparableConvolution_updateOutput_backward_vertical1<<< (n_vertical + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
		    n_vertical,
		    THCudaTensor_data(state, input), make_long4(input->size[0], input->size[1], input->size[2], input->size[3]), make_long4(input->stride[0], input->stride[1], input->stride[2], input->stride[3]),
		    THCudaTensor_data(state, horizontal), make_long4(horizontal->size[0], horizontal->size[1], horizontal->size[2], horizontal->size[3]), make_long4(horizontal->stride[0], horizontal->stride[1], horizontal->stride[2], horizontal->stride[3]),
		    THCudaTensor_data(state, grad_output), make_long4(grad_output->size[0], grad_output->size[1], grad_output->size[2], grad_output->size[3]), make_long4(grad_output->stride[0], grad_output->stride[1], grad_output->stride[2], grad_output->stride[3]),
		    THCudaTensor_data(state, grad_vertical), make_long4(grad_vertical->size[0], grad_vertical->size[1], grad_vertical->size[2], grad_vertical->size[3]), make_long4(grad_vertical->stride[0], grad_vertical->stride[1], grad_vertical->stride[2], grad_vertical->stride[3])
        );
 
        //horizontal gradient
	    kernel_SeparableConvolution_updateOutput_backward_horizontal1<<<(n_horizontal + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
		    n_horizontal,
		    THCudaTensor_data(state, input), make_long4(input->size[0], input->size[1], input->size[2], input->size[3]), make_long4(input->stride[0], input->stride[1], input->stride[2], input->stride[3]),
		    THCudaTensor_data(state, vertical), make_long4(vertical->size[0], vertical->size[1], vertical->size[2], vertical->size[3]), make_long4(vertical->stride[0], vertical->stride[1], vertical->stride[2], vertical->stride[3]),
		    THCudaTensor_data(state, grad_output), make_long4(grad_output->size[0], grad_output->size[1], grad_output->size[2], grad_output->size[3]), make_long4(grad_output->stride[0], grad_output->stride[1], grad_output->stride[2], grad_output->stride[3]),
		    THCudaTensor_data(state, grad_horizontal), make_long4(grad_horizontal->size[0], grad_horizontal->size[1], grad_horizontal->size[2], grad_horizontal->size[3]), make_long4(grad_horizontal->stride[0], grad_horizontal->stride[1], grad_horizontal->stride[2], grad_horizontal->stride[3])
        ); 
    };

	THCudaCheck(cudaGetLastError());
}


void SeparableConvolution_kernel_backward_only_vertical(
	THCState* state,
	THCudaTensor* input,
	THCudaTensor* grad_output,
	THCudaTensor* grad_vertical
) {
	int n_vertical = 0;
	n_vertical = THCudaTensor_nElement(state, grad_vertical);

        
    //vertical gradient
	kernel_SeparableConvolution_updateOutput_backward_vertical0<<< (n_vertical + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
		    n_vertical,
		    THCudaTensor_data(state, input), make_long4(input->size[0], input->size[1], input->size[2], input->size[3]), make_long4(input->stride[0], input->stride[1], input->stride[2], input->stride[3]),
		    THCudaTensor_data(state, grad_output), make_long4(grad_output->size[0], grad_output->size[1], grad_output->size[2], grad_output->size[3]), make_long4(grad_output->stride[0], grad_output->stride[1], grad_output->stride[2], grad_output->stride[3]),
		    THCudaTensor_data(state, grad_vertical), make_long4(grad_vertical->size[0], grad_vertical->size[1], grad_vertical->size[2], grad_vertical->size[3]), make_long4(grad_vertical->stride[0], grad_vertical->stride[1], grad_vertical->stride[2], grad_vertical->stride[3])
    );

	THCudaCheck(cudaGetLastError());
}

void SeparableConvolution_kernel_backward_only_horizontal(
	THCState* state,
	THCudaTensor* input,
	THCudaTensor* grad_output,
	THCudaTensor* grad_horizontal
) {
	int n_horizontal = 0;
	n_horizontal = THCudaTensor_nElement(state, grad_horizontal);

        
    //vertical gradient
	kernel_SeparableConvolution_updateOutput_backward_horizontal0<<< (n_horizontal + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
		    n_horizontal,
		    THCudaTensor_data(state, input), make_long4(input->size[0], input->size[1], input->size[2], input->size[3]), make_long4(input->stride[0], input->stride[1], input->stride[2], input->stride[3]),
		    THCudaTensor_data(state, grad_output), make_long4(grad_output->size[0], grad_output->size[1], grad_output->size[2], grad_output->size[3]), make_long4(grad_output->stride[0], grad_output->stride[1], grad_output->stride[2], grad_output->stride[3]),
		    THCudaTensor_data(state, grad_horizontal), make_long4(grad_horizontal->size[0], grad_horizontal->size[1], grad_horizontal->size[2], grad_horizontal->size[3]), make_long4(grad_horizontal->stride[0], grad_horizontal->stride[1], grad_horizontal->stride[2], grad_horizontal->stride[3])
    );

	THCudaCheck(cudaGetLastError());
}


#ifdef __cplusplus
	}
#endif
