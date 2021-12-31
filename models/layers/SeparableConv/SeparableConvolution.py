import torch
import PIL
import numpy as np
import _ext.cunnex

class SeparableConvolution(torch.autograd.Function):
    def __init__(self, filter_size_vertical, filter_size_horizontal):
        self.g_input = None
        self.g_vertical = None
        self.g_horizontal = None
        self.filter_size_vertical = filter_size_vertical
        self.filter_size_horizontal = filter_size_horizontal
        super(SeparableConvolution, self).__init__()
    # end

    def forward(self, input, vertical, horizontal):
        self.g_input = input
        self.g_vertical = vertical
        self.g_horizontal = horizontal
        intBatches = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSizeVertical = vertical.size(1)
        intFilterSizeHorizontal = horizontal.size(1)
        intOutputHeight = min(vertical.size(2), horizontal.size(2))
        intOutputWidth = min(vertical.size(3), horizontal.size(3))

        #Code for printing horizontal kernels
        #print(horizontal[:,:,100:105,150])
        '''
        np_horizontal = horizontal.cpu().float().numpy()

        out_horizontal = np_horizontal[0,:,:,100]
        out_horizontal -= np.min(out_horizontal)
        out_horizontal /= np.max(out_horizontal)
        out_horizontal *= 255
        PIL.Image.fromarray(out_horizontal.astype(np.uint8)).save("horizontal.jpg")

        del np_horizontal
        del out_horizontal
        '''
        #print np.max(horizontal[0,:,200,100])
        #print(horizontal[0,:,200,150])
        #print(horizontal[0,:,345,150])
        #print(horizontal[0,:,345,750])
        #print(horizontal[0,:,134,750])
        #exit()
        #print(intInputHeight)
        if(self.filter_size_vertical >= 1):
            assert(intInputHeight - self.filter_size_vertical == intOutputHeight - 1)
            assert(intFilterSizeVertical == self.filter_size_vertical)
        if(self.filter_size_horizontal >= 1):
            assert(intInputWidth - self.filter_size_horizontal == intOutputWidth - 1)
            assert(intFilterSizeHorizontal == self.filter_size_horizontal)

        assert(input.is_contiguous() == True)
        assert(vertical.is_contiguous() == True)
        assert(horizontal.is_contiguous() == True)

        output = input.new().resize_(intBatches, intInputDepth, intOutputHeight, intOutputWidth).zero_()
        if input.is_cuda == True:
            _ext.cunnex.SeparableConvolution_cuda_forward(
                    input,
                    vertical,
                    self.filter_size_vertical,
                    horizontal,
                    self.filter_size_horizontal,
                    output
                    )

        elif input.is_cuda == False:
            raise NotImplementedError() # CPU VERSION NOT IMPLEMENTED
        # end
        return output
    # end

    # end
    #@staticmethod
    def backward(self, grad_output):
        input = self.g_input
        vertical = self.g_vertical
        horizontal = self.g_horizontal
        #print(vertical)
        #exit()
        intBatches = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSizeVertical = vertical.size(1)
        intFilterSizeHorizontal = horizontal.size(1)
        intOutputHeight = min(vertical.size(2), horizontal.size(2))
        intOutputWidth = min(vertical.size(3), horizontal.size(3))

        grad_input = input.new().resize_(intBatches, intInputDepth, intInputHeight, intInputWidth).zero_()
        grad_vertical = vertical.new().resize_(intBatches, intFilterSizeVertical, intOutputHeight, intOutputWidth).zero_()
        grad_horizontal = horizontal.new().resize_(intBatches, intFilterSizeHorizontal, intOutputHeight, intOutputWidth).zero_()

        #print(input.size())
        #print(grad_output.size())
        #print(intOutputHeight)
        #print(vertical.size())
        #print('start backward')
        #print(grad_output)
        #grad_output = grad_output.cuda()
        #print(grad_output.min(), grad_output.max())
        #exit()

        if grad_output.is_cuda == True:
            _ext.cunnex.SeparableConvolution_cuda_backward(
                    input,
                    vertical,
                    self.filter_size_vertical,
                    horizontal,
                    self.filter_size_horizontal,
                    grad_output,
                    grad_vertical,
                    grad_horizontal
                    )


        elif grad_output.is_cuda == False:
            raise NotImplementedError() # CPU VERSION NOT IMPLEMENTED

        #print(grad_horizontal[:,:,200:205,450])

        return grad_input, grad_vertical, grad_horizontal

    def gradient_check():

        input = torch.autograd.Variable(torch.randn(1,3,3,5).cuda(),requires_grad=True)
        vertical = torch.autograd.Variable(torch.randn(1,1,3,3).cuda(),requires_grad=True) # Vertical Kernel Size
        horizontal = torch.autograd.Variable(torch.randn(1,3,3,3).cuda(),requires_grad=True) # Horizontal kernel size

        print(torch.autograd.gradcheck(SeparableConvolution(),(input,vertical,horizontal),eps=1e-6,atol=1,raise_exception=True))

    #print(input.size())
        #print(grad_vertical.size())
        '''
        #print(horizontal.size())

        vertical = vertical.cpu().float().numpy()
        horizontal = horizontal.cpu().float().numpy()
        grad_vertical = grad_vertical.cpu().float().numpy()
        grad_horizontal = grad_horizontal.cpu().float().numpy()

        #print(np.max(vertical))
        grad_output = grad_output.cpu().numpy()
        input = input.cpu().float().numpy()

        #print(np.shape(input[:,0,1:1+51,1:1+51]))
        # This is wrong implementation
        for m in range(intOutputHeight):
            for n in range(intOutputWidth):
                #result = np.multiply(horizontal[:,:,m,n],(input[:,0,m:m+51,n:n+51]+input[:,1,m:m+51,n:n+51]+input[:,2,m:m+51,n:n+51])/3)/51
                #print(np.shape(result))
                #print(grad_output.size())
                #print(np.shape(grad_vertical))
                #exit()

                vertical_result = np.dot(horizontal[:,:,m,n],(input[:,0,m:m+51,n:n+51]*grad_output[:,0,m,n] + input[:,1,m:m+51,n:n+51]*grad_output[:,1,m,n] + input[:,2,m:m+51,n:n+51]*grad_output[:,2,m,n]).transpose())
                horizontal_result = np.dot(vertical[:,:,m,n],(input[:,0,m:m+51,n:n+51]*grad_output[:,0,m,n] + input[:,1,m:m+51,n:n+51]*grad_output[:,1,m,n] + input[:,2,m:m+51,n:n+51]*grad_output[:,2,m,n]))
                grad_vertical[:,:,m,n] = vertical_result[:,:,0]
                grad_horizontal[:,:,m,n] = horizontal_result[:,:,0].transpose()
        #print(np.max(grad_vertical))
        #exit()

        #print(type(torch.from_numpy(grad_vertical)))
        #exit()
        #return grad_input.cuda(), torch.from_numpy(grad_vertical).cuda(), torch.from_numpy(grad_horizontal).cuda()
        #print(input)
        #exit()


        return grad_input, torch.from_numpy(grad_vertical).cuda(), torch.from_numpy(grad_horizontal).cuda()

        input = torch.autograd.Variable(torch.randn((1,3,3,5).cuda()),requires_grad=True)
        vertical = torch.autograd.Variable(torch.randn((1,1,3,3).cuda()),requires_grad=True) # Vertical Kernel Size
        horizontal = torch.autograd.Variable(torch.randn((1,3,3,3).cuda()),requires_grad=True) # Horizontal kernel size

        print(torch.autograd.gradcheck(SeparableConvolution(1,3),(input,vertical,horizontal),eps=1e-6,atol=1,raise_exception=True))
        '''
    # end
    # end
 # BACKPROPAGATION NOT IMPLEMENTED
    # end
# end
