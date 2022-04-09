
import torch 

tensor = torch.ones(4, 4)
# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
#tensor[:,1] = 0 # All ROW's and Column -1 to ZERO's 
tensor[:,2] = 22 # All ROW's and Column -2 to 22 
print(tensor)

"""
First row: tensor([1., 1., 1., 1.], device='cuda:0')
First column: tensor([1., 1., 1., 1.], device='cuda:0')
Last column: tensor([1., 1., 1., 1.], device='cuda:0')

### tensor[:,1] = 0 # All ROW's and Column -1 to ZERO's 

tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]], device='cuda:0')
"""
#
"""
First row: tensor([1., 1., 1., 1.], device='cuda:0')
First column: tensor([1., 1., 1., 1.], device='cuda:0')
Last column: tensor([1., 1., 1., 1.], device='cuda:0')

### tensor[:,2] = 22 # All ROW's and Column -2 to 22

tensor([[ 1.,  1., 22.,  1.],
        [ 1.,  1., 22.,  1.],
        [ 1.,  1., 22.,  1.],
        [ 1.,  1., 22.,  1.]], device='cuda:0')
"""
#
#Joining tensors 
# You can use torch.cat to concatenate a sequence of tensors along a given dimension. 
# See also torch.stack, another tensor joining op that is subtly different from torch.cat.
print("--tensor.shape----\n",tensor.shape) # torch.Size([4, 4])

t1 = torch.cat([tensor, tensor], dim=1)

print("--t1.shape----\n",t1.shape) # torch.Size([4, 8])
print(t1)
#

