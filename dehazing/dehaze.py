import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dehazing.dataloader as dataloader
import dehazing.net as net
import numpy as np
from torchvision import transforms
from PIL import Image
import glob

dehaze_net = net.dehaze_net()
dehaze_net.load_state_dict(torch.load('dehazing/snapshots/dehazer.pth', map_location=torch.device('mps')))

def dehaze(img):
	data_hazy = torch.from_numpy(img).float()
	data_hazy = data_hazy.permute(2,0,1)
	data_hazy = data_hazy.unsqueeze(0)

	tensor = dehaze_net(data_hazy)

	# バッチ次元を取り除くためにテンソルをsqueeze
	tensor = tensor.squeeze(0)  # (チャネル, 高さ, 幅)

	# テンソルをCPU上に移動させ、NumPy配列に変換
	numpy_array = tensor.cpu().detach().numpy()  # (チャネル, 高さ, 幅)

	# チャネルの次元を最後に移動
	numpy_array = np.transpose(numpy_array, (1, 2, 0))  # (高さ, 幅, チャネル)

	return numpy_array
	

def dehaze_image(image_path):

	data_hazy = Image.open(image_path)
	data_hazy = (np.asarray(data_hazy)/255.0)

	data_hazy = torch.from_numpy(data_hazy).float()
	data_hazy = data_hazy.permute(2,0,1)
	data_hazy = data_hazy.unsqueeze(0)

	dehaze_net = net.dehaze_net()
	dehaze_net.load_state_dict(torch.load('snapshots/dehazer.pth', map_location=torch.device('ops')))

	clean_image = dehaze_net(data_hazy)
	torchvision.utils.save_image(torch.cat((data_hazy, clean_image),0), "results/" + image_path.split("/")[-1])
	

if __name__ == '__main__':

	test_list = glob.glob("test_images/*")

	for image in test_list:

		dehaze_image(image)
		print(image, "done!")
