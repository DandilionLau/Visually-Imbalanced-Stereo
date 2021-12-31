#include <THC.h>
#include <THCGeneral.h>

#include "SeparableConvolution_kernel.h"

extern THCState* state;

int SeparableConvolution_cuda_forward(
	THCudaTensor* input,
	THCudaTensor* vertical,
    int size_vertical,
	THCudaTensor* horizontal,
    int size_horizontal,
	THCudaTensor* output
) {
	
    if(size_vertical > 0) {
        
        if(size_horizontal > 0) {
            SeparableConvolution_kernel_forward(
		        state,
		        input,
		        vertical,
		        horizontal,
		        output
	        );
        } else if (size_horizontal == 0) {
            SeparableConvolution_kernel_forward_only_vertical(
		        state,
		        input,
		        vertical,
		        output
	        );
        } else {
            //Invalid arguments: size_horizontal < 0
            return -1;            
        }

    } else if(size_vertical == 0) {
        
        if(size_horizontal > 0) {
            SeparableConvolution_kernel_forward_only_horizontal(
		        state,
		        input,
		        horizontal,
		        output
	        );
        } else {
            //Invalid arguments:
            return -1;
        }
        
    } else {
        
        //Invalid arguments
        return -1;
    }

	return 1;
}

int SeparableConvolution_cuda_backward(
	THCudaTensor* input,
	THCudaTensor* vertical,
    int size_vertical,
	THCudaTensor* horizontal,
    int size_horizontal,
	THCudaTensor* grad_output,
	THCudaTensor* grad_vertical,
	THCudaTensor* grad_horizontal
) {
    
    if(size_vertical > 0) {
        
        if(size_horizontal > 0) {
	        SeparableConvolution_kernel_backward(
		        state,
		        input,
		        vertical,
		        horizontal,
		        grad_output,
                grad_vertical,
                grad_horizontal
	        );
        
        } else if (size_horizontal == 0) {
            SeparableConvolution_kernel_backward_only_vertical(
		        state,
		        input,
		        grad_output,
		        grad_vertical
	        );
        } else {
            //Invalid arguments: size_horizontal < 0
            return -1;            
        }

    } else if(size_vertical == 0) {
        
        if(size_horizontal > 0) {
            SeparableConvolution_kernel_backward_only_horizontal(
		        state,
		        input,
		        grad_output,
		        grad_horizontal
	        );
        } else {
            //Invalid arguments:
            return -1;
        }
        
    } else {
        
        //Invalid arguments
        return -1;
    }



	return 1;
}


