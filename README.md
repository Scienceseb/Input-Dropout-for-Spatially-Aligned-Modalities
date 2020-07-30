# Input-Dropout
PyTorch code for the paper **Input Dropout for Spatially Aligned Modalities** (https://arxiv.org/pdf/2002.02852.pdf)<br/> 

Two assumptions:<br/>
1) All input modalities are spatially aligned (that must be true).<br/>
2) RGB modality is the only modality available at test time (this assumption is for the paper only, you can make the change in the code).

## **The approach:**<br/>
The additional modality is first channel-wise concatenated to the RGB image, and the resulting tensor is fed as input to the neural network. The first convolutional layer of the network must be adapted to this new input dimensionality. At training time, one of the input modalities is randomly set to 0 with probability between 0 and 1. This effectively “drops out” the corresponding modality. At test time, the additional modality is always set to 0. Since we assume a single additional modality is combined with an RGB image, we are faced with two options. We could randomly drop only the additional modality and always keep the RGB (we dub this option addit), or drop either the RGB or the additional modality (we dub this option both). In these two cases, a uniform probability distribution for the different possible cases is used.

## **How can I put Input Dropout in my problem? (InputDropout.py)**<br/>
To use the proposed method, simply take the code block from the InputDropout.py file named InputDropout. This block is used exactly like a PyTorch transform function. As an argument, just put the drop mode (either both or addit). It is very easy to modify this code if you want to experiment with two additional modalities for example. The limits are your imagination.


## **Easy example, showing how to use Input Dropout, but not really improving performance (main_cifar10.py):** <br/>
A simple example of how to use this code is as follows: take CIFAR-10, duplicate the dataset and make the copy black and white (BW). So now we have RGB and BW pairs. We just have to concatenate these pairs to obtain a 4-channel RGB-BW image. In this example, you can easily test the different Input Dropout modes. See the code for more details. 

## **The poster:** <br/>
![Poster](poster.png)
