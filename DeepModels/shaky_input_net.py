'''
this is an experiment about shaky input training method. In this network we introduce a batch
of data which contains the same information in each layer with a slight perturbation added to the 
each layer. this should improve the focus of the network.
as much as the variance increases inside the same batch we penalize the model in the backpropagaion
'''