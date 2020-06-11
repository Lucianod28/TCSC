import argparse


parser = argparse.ArgumentParser(description="Template")
# model
parser.add_argument('-B', '--batch_size', default=2000, type=int, help="Batch size")
parser.add_argument('-N', '--n_neuron', default=400, type=int, help="The number of neurons")
parser.add_argument('-M', '--size', default=10, type=int, help="The size of receptive field")
parser.add_argument('-K', '--kernel_size', default=5, type=int, help="The size of the kernel for transposed convolution")
parser.add_argument('-S', '--stride', default=1, type=int, help="The stride for the transposed convolution")
parser.add_argument('-T', '--train_conv', action='store_true', help="Whether to train the conv_trans layer")
# training
parser.add_argument('-e', '--epoch', default=100, type=int, help="Number of Epochs")
parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float, help="Learning rate")
parser.add_argument('-rlr', '--r_learning_rate', default=1e-2, type=float, help="Learning rate for ISTA")
parser.add_argument('-lmda', '--reg', default=5e-3, type=float, help="LSTM hidden size")


# Parse arguments
def parse_args():
	return parser.parse_args()
