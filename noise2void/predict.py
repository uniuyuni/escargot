import numpy as np

from noise2void.model import *
import noise2void.splitimage as splitimage

def setup_predict(gpu_ids=[0], nch_in=3, nch_out=3, nch_ker=64, norm='bnorm'):

    ## setup network
    # netG = UNet(nch_in, nch_out, nch_ker, norm)
    netG = ResNet(nch_in, nch_out, nch_ker, norm)
    init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

    epoch = 300
    dict_net = torch.load('./noise2void/checkpoint/resnet/bsds500/model_epoch%04d.pth' % epoch)
    netG.load_state_dict(dict_net['netG'])
    print('Loaded %dth network' % epoch)

    return netG

def predict(image, netG, device):

    tdevice = torch.device(device)

    with torch.no_grad():
        netG.eval()

        print("split_image_with_overlap")
        input = image
        input = (input - 0.5) / 0.5
        input_blocks, split_info = splitimage.split_image_with_overlap(input, 1024, 1024, 32)
        print(split_info)

        output_blocks = []
        for n, block in enumerate(input_blocks, 0):
            print(n, block.shape)
            _in = np.pad(block, ((0,0), (4,0), (0,0)), mode='edge')
            _in = torch.from_numpy(_in.transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0).to(tdevice)
            _out = netG(_in)
            _out = _out.to('cpu').detach().numpy().transpose(0, 2, 3, 1)[0,:,:,:].squeeze()
            _out = np.delete(_out, [0, 1, 2, 3], 1)
            output_blocks.append(_out)
        
        print("combine_image_with_overlap")         
        output = splitimage.combine_image_with_overlap(output_blocks, split_info)
        output = output*0.5+0.5
        output = np.clip(output, 0., 1.)
        
        return output
