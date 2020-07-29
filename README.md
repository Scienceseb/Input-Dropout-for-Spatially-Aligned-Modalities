# Input-Dropout
PyTorch code for the paper Input Dropout for Spatially Aligned Modalities (https://arxiv.org/pdf/2002.02852.pdf) 

Two assumptions:
1) All input modalities are spatially aligned.
2) RGB modality is the only modality available at test time.

The approach:
The additional modality is first channel-wise concatenated to the RGB image, and the resulting tensor is fed as input to the neural network. The first convolutional layer of the network must be adapted to this new input dimensionality. At training time, one of the input modalities is randomly set to 0 with probability between 0 and 1. This effectively “drops out” the corresponding modality. At test time, the additional modality is always set to 0. Since we assume a single additional modality is combined with an RGB image, we are faced with two options. We could randomly drop only the additional modality and always keep the RGB (we dub this option addit), or drop either the RGB or the additional modality (we dub this option both). In these two cases, a uniform probability distribution for the different possible cases is used.

How do I put Input Dropout in my problem?
To use the proposed method, simply take the code block from the ID.py file named Input Dropout. This block is used exactly like a PyTorch transform function. As an argument, just put the drop mode (either both or addit).
 
