## SOURCE >> Python Engineer  >> https://www.youtube.com/watch?v=DbeIqrwb_dE&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=3
## PyTorch Tutorial 03 - Gradient Calculation With Autograd
## 


import torch 

torch.manual_seed(1)
#
x = torch.randn(3)
print(x)
y = x + 30
print(y)
#
x_1 = torch.randn(3,requires_grad=True)
print(x_1)
#
y_1 = x_1 + 30
#y_1 = y_1(requires_grad=True)
print(y_1) #tensor([30.6213, 29.5481, 29.8339], grad_fn=<AddBackward0>)
#
"""
tensor([0.6614, 0.2669, 0.0617])
tensor([ 0.6213, -0.4519, -0.1661], requires_grad=True)
tensor([30.6213, 29.5481, 29.8339], grad_fn=<AddBackward0>)
"""
#
z_1 = y_1 * y_1 * 2
print(z_1)
#
"""
tensor([0.6614, 0.2669, 0.0617])
tensor([30.6614, 30.2669, 30.0617])
tensor([ 0.6213, -0.4519, -0.1661], requires_grad=True)
tensor([30.6213, 29.5481, 29.8339], grad_fn=<AddBackward0>)
tensor([1875.3301, 1746.1797, 1780.1196], grad_fn=<MulBackward0>)
"""
#
#z_1 = z_1.mean() ## need a test_vector == [0.1 , 0.11 , 0.111]
test_vector = torch.tensor([0.1 , 0.11 , 0.111])
z_1.backward(test_vector)
print("x_1.grad---\n",x_1.grad)
#print(y_1.grad)
#
# ### below with -- z_1.backward() ## NO PARAM
# tensor([0.6614, 0.2669, 0.0617])
# tensor([30.6614, 30.2669, 30.0617])
# tensor([ 0.6213, -0.4519, -0.1661], requires_grad=True)
# tensor([30.6213, 29.5481, 29.8339], grad_fn=<AddBackward0>)
# tensor([1875.3301, 1746.1797, 1780.1196], grad_fn=<MulBackward0>)
# x_1.grad---
#  tensor([40.8284, 39.3975, 39.7785])

### below with -- z_1.backward(test_vector)
# tensor([0.6614, 0.2669, 0.0617])
# tensor([30.6614, 30.2669, 30.0617])
# tensor([ 0.6213, -0.4519, -0.1661], requires_grad=True)
# tensor([30.6213, 29.5481, 29.8339], grad_fn=<AddBackward0>)
# tensor([1875.3301, 1746.1797, 1780.1196], grad_fn=<MulBackward0>)
# x_1.grad---
#  tensor([12.2485, 13.0012, 13.2462])

#
"""
#Below Error if we dont get the MEAN of the z_1 ....
#As when we take he MEAN of the z_1 -- we get a SCALAR value 
#when z_1 is a scalar - we can call .backward() ... without any PARAMS
#but if its not s scalar - then we will need to pass in a VECTOR with same SHAPE as the -- non scalar z_1 
# same shape vector will be -- test_vector == [0.1 , 0.11 , 0.111]


File "test_2.py", line 33, in <module>
    z_1.backward()
  File "/home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages/torch/_tensor.py", line 363, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)
  File "/home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages/torch/autograd/__init__.py", line 166, in backward
    grad_tensors_ = _make_grads(tensors, grad_tensors_, is_grads_batched=False)
  File "/home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages/torch/autograd/__init__.py", line 67, in _make_grads
    raise RuntimeError("grad can be implicitly created only for scalar outputs")
RuntimeError: grad can be implicitly created only for scalar outputs
"""
#
