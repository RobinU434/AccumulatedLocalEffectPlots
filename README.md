# AccumulatedLocalEffectPlots
This repository is part of the "Fair and Interpretable Machine Learning" seminar hosted by Janek Thomas in the summer semester 2022. This codebase will contain a personal custom implementation of accumulated effect plots.
 
I will use out of the box ML algorithms to generate models to apply the presented interpretability technique. 

For initialization you can download the required datasets with the provided script.

## Application on images
an application of ALE plots on images could be very difficult to understand since we are applying the method on every single pixel but the actual meaning of an image is the result of almost every single pixel and their constalation. It is also very costly since we have to do calculate a plot for every single pixel or their combinations. 
Here I would propose to apply ALE plots after the output of a particular Kernel. We are able to find out on which feature the kernel is focusing on (straight line or circles or ...) and then apply the ALE plot to actually find out with what margin this kernel output is contributing to the actual output of the whole neuronal network.