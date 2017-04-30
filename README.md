# Convolutional-Neural-Network
A convolutional neural network for image classification with the CIFAR-10 dataset

This is now a fully functioning convolutional network.
In between this and the "regular" neural network I made another convolutional network
with a 3d volume container class, but it ended up being horrendously slow.
Therefore this version is a bit stripped down, i went with just good-old arrays
and switched to floats  to speed it up, as well as just avoiding as much overhead as possible.

The things that have been added that are not in "Neural Network" are:
 - A convoluitonal layer that does zero-padding by default, so it preserves spatial size unless stride > 1 is used
 - Max pooling layer that does a filter size 2, stride 2 max pool, to just downsample by half
 
The current setup managed to classify just over 70% correctly on the entire validation set after having trained for 10 epochs. 10 epochs still took almost 2 hours, so it's still pretty slow. I'm currently looking into CUDA as a means of speeding it up further, I probably won't do much more with this project until i can speed things up properly in gpu.
