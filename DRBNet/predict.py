
from DRBNet.models.DRBNet import *
import DRBNet.splitimage as splitimage

def setup_predict():
    network = DRBNet_single().to('mps')
    network.load_state_dict(torch.load('DRBNet/ckpts/single/single_image_defocus_deblurring.pth'))

    return network

def predict(input, network, device):
    network.eval()

    input = input*2.0-1.0
    input_blocks, split_info = splitimage.split_image_with_overlap(input, 1024, 1024, 32)

    with torch.no_grad():
        output_blocks = []
        for n, block in enumerate(input_blocks, 0):
            print(n, block.shape)
            block = np.expand_dims(block,  axis = 0)
            C = torch.FloatTensor(block.transpose(0, 3, 1, 2).copy()).to(device)
            output = network(C)
            output_cpu = (output.cpu().numpy()[0].transpose(1, 2, 0) +1.0 )*0.5
            output_blocks.append(output_cpu)
    
    output = splitimage.combine_image_with_overlap(output_blocks, split_info)
    
    return output