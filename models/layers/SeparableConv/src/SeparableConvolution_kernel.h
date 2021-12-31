#ifdef __cplusplus
	extern "C" {
#endif

void SeparableConvolution_kernel_forward(
	THCState* state,
	THCudaTensor* input,
	THCudaTensor* vertical,
	THCudaTensor* horizontal,
	THCudaTensor* output
);

void SeparableConvolution_kernel_forward_only_vertical(
	THCState* state,
	THCudaTensor* input,
	THCudaTensor* vertical,
	THCudaTensor* output
);

void SeparableConvolution_kernel_forward_only_horizontal(
	THCState* state,
	THCudaTensor* input,
	THCudaTensor* horizontal,
	THCudaTensor* output
);

void SeparableConvolution_kernel_backward(
	THCState* state,
	THCudaTensor* input,
	THCudaTensor* vertical,
	THCudaTensor* horizontal,
	THCudaTensor* grad_output,
	THCudaTensor* grad_vertical,
	THCudaTensor* grad_horizontal
);

void SeparableConvolution_kernel_backward_only_vertical(
	THCState* state,
	THCudaTensor* input,
	THCudaTensor* grad_output,
	THCudaTensor* grad_vertical
);

void SeparableConvolution_kernel_backward_only_horizontal(
	THCState* state,
	THCudaTensor* input,
	THCudaTensor* grad_output,
	THCudaTensor* grad_horizontal
);

#ifdef __cplusplus
	}
#endif
