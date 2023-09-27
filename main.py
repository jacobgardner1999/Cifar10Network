import network
import cifar_loader

training_data, test_data = cifar_loader.load_data()
net = network.Network([3072, 30, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
