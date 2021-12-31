int SeparableConvolution_cuda_forward(
	THCudaTensor* input,
	THCudaTensor* vertical,
    int size_vertical,
	THCudaTensor* horizontal,
    int size_horizontal,
	THCudaTensor* output
);

int SeparableConvolution_cuda_backward(
	THCudaTensor* input,
	THCudaTensor* vertical,
    int size_vertical,
	THCudaTensor* horizontal,
    int size_horizontal,
	THCudaTensor* grad_output,
	THCudaTensor* grad_vertical,
	THCudaTensor* grad_horizontal
);
