
#

#### AWESOME -- SEMANTIC SEGMENTATION 

- https://github.com/mrgloom/awesome-semantic-segmentation


#

#### See model Summary 

#
- https://stackoverflow.com/questions/42480111/model-summary-in-pytorch

- Yes, you can get exact Keras representation, using the pytorch-summary package.

```python

from torchsummary import summary
summary(your_model, input_size=(channels, H, W))

```

```pthon
from torchvision import models
from torchsummary import summary

vgg = models.vgg16()
summary(vgg, (3, 224, 224))
```

- own error 

```

Traceback (most recent call last):
  File "get_kaggle_data.py", line 114, in <module>
    model_summary = summary(model, input_size=(3, 800, 800))
  File "/home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages/torchsummary/torchsummary.py", line 72, in summary
    model(*x)
  File "/home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages/torchvision/models/detection/generalized_rcnn.py", line 59, in forward
    raise ValueError("In training mode, targets should be passed")
ValueError: In training mode, targets should be passed
(pytorch_venv) dhankar@dhankar-1:~/.../_book_4$ 

```
#
- https://github.com/sksq96/pytorch-summary

- https://stackoverflow.com/questions/60484859/pytorch-why-printmodel-does-not-show-the-activation-functions

- FOOBAR -- he means add the FUNCTIONAL PROGRAMMING or the OBJECT -- OOPS way 
- There are two ways of adding operations to the network graph: lowlevel functional way and more advanced object way. 

#

- Pytorch Model Summary = https://stackoverflow.com/questions/66830798/pytorch-model-summary

- You loaded the "*.pt" and didn't feed it to a model (which is just a dictionary of the weights depending on what you saved)

#
