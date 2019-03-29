'''
this is an experiment unstable weights network, the weights are randomly and slightly perturbated
during training and best results are kept after each lap.
the lap is made of several inference rounds which before to be made we apply perturbation
over the weights.
we keep track of the tendancy of the network after each round to monitor the accuracy
'''