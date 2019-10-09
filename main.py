# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:00:32 2019

@author: Mozhdeh
"""

import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784,30,10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)