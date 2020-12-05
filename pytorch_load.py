import os

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
from nasnet import NASNetALarge, NASNetAMobile
#from models.nasnet.debug import read_output


######################################################################
## Load parameters from HDF5 to Dict
######################################################################
#from models.util import save_model


def load_conv2d(state_dict, path, name_pth, name_tf):
    h5f = h5py.File(path + '/' + name_tf + '.h5', 'r')
    state_dict[name_pth + '.weight'] = torch.from_numpy(h5f['weight'][()]).permute(3, 2, 0, 1)
    try:
        state_dict[name_pth + '.bias'] = torch.from_numpy(h5f['bias'][()])
    except:
        pass
    h5f.close()


def load_linear(state_dict, path, name_pth, name_tf):
    h5f = h5py.File(path + '/' + name_tf + '.h5', 'r')
    state_dict[name_pth + '.weight'] = torch.from_numpy(h5f['weight'][()]).t()
    try:
        state_dict[name_pth + '.bias'] = torch.from_numpy(h5f['bias'][()])
    except:
        pass
    h5f.close()


def load_bn(state_dict, path, name_pth, name_tf):
    h5f = h5py.File(path + '/' + name_tf + '.h5', 'r')
    state_dict[name_pth + '.weight'] = torch.from_numpy(h5f['gamma'][()])
    state_dict[name_pth + '.bias'] = torch.from_numpy(h5f['beta'][()])
    state_dict[name_pth + '.running_mean'] = torch.from_numpy(h5f['mean'][()])
    state_dict[name_pth + '.running_var'] = torch.from_numpy(h5f['var'][()])
    h5f.close()


def load_separable_conv2d(state_dict, path, name_pth, name_tf):
    h5f = h5py.File(path + '/' + name_tf + '.h5', 'r')
    state_dict[name_pth + '.depthwise_conv2d.weight'] = torch.from_numpy(h5f['depthwise_weight'][()]).permute(2, 3, 0,
                                                                                                              1)
    try:
        state_dict[name_pth + '.depthwise_conv2d.bias'] = torch.from_numpy(h5f['depthwise_bias'][()])
    except:
        pass
    state_dict[name_pth + '.pointwise_conv2d.weight'] = torch.from_numpy(h5f['pointwise_weight'][()]).permute(3, 2, 0,
                                                                                                              1)
    try:
        state_dict[name_pth + '.pointwise_conv2d.bias'] = torch.from_numpy(h5f['pointwise_bias'][()])
    except:
        pass
    h5f.close()


def load_cell_branch(state_dict, path, name_pth, name_tf, branch, kernel_size):
    load_separable_conv2d(state_dict, path, name_pth=name_pth + '_{branch}.separable_1'.format(branch=branch),
                          name_tf=name_tf + '/{branch}/separable_{ks}x{ks}_1'.format(branch=branch, ks=kernel_size))
    load_bn(state_dict, path, name_pth=name_pth + '_{branch}.bn_sep_1'.format(branch=branch),
            name_tf=name_tf + '/{branch}/bn_sep_{ks}x{ks}_1'.format(branch=branch, ks=kernel_size))
    load_separable_conv2d(state_dict, path, name_pth=name_pth + '_{branch}.separable_2'.format(branch=branch),
                          name_tf=name_tf + '/{branch}/separable_{ks}x{ks}_2'.format(branch=branch, ks=kernel_size))
    load_bn(state_dict, path, name_pth=name_pth + '_{branch}.bn_sep_2'.format(branch=branch),
            name_tf=name_tf + '/{branch}/bn_sep_{ks}x{ks}_2'.format(branch=branch, ks=kernel_size))


def load_cell_stem_0(state_dict, path, name_pth='cell_stem_0', name_tf='cell_stem_0'):
    # conv 1x1
    load_conv2d(state_dict, path, name_pth=name_pth + '.conv_1x1.conv', name_tf=name_tf + '/1x1')
    load_bn(state_dict, path, name_pth=name_pth + '.conv_1x1.bn', name_tf=name_tf + '/beginning_bn')

    # comb_iter_0
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_0', name_tf=name_tf + '/comb_iter_0',
                     branch='left', kernel_size=5)
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_0', name_tf=name_tf + '/comb_iter_0',
                     branch='right', kernel_size=7)

    # comb_iter_1
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_1', name_tf=name_tf + '/comb_iter_1',
                     branch='right', kernel_size=7)

    # comb_iter_2
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_2', name_tf=name_tf + '/comb_iter_2',
                     branch='right', kernel_size=5)

    # comb_iter_4
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_4', name_tf=name_tf + '/comb_iter_4',
                     branch='left', kernel_size=3)


def load_cell_stem_1(state_dict, path, name_pth='cell_stem_1', name_tf='cell_stem_1'):
    # conv 1x1
    load_conv2d(state_dict, path, name_pth=name_pth + '.conv_1x1.conv', name_tf=name_tf + '/1x1')
    load_bn(state_dict, path, name_pth=name_pth + '.conv_1x1.bn', name_tf=name_tf + '/beginning_bn')

    load_conv2d(state_dict, path, name_pth=name_pth + '.path_1.conv', name_tf=name_tf + '/path1_conv')
    load_conv2d(state_dict, path, name_pth=name_pth + '.path_2.conv', name_tf=name_tf + '/path2_conv')
    load_bn(state_dict, path, name_pth=name_pth + '.final_path_bn', name_tf=name_tf + '/final_path_bn')

    # comb_iter_0
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_0', name_tf=name_tf + '/comb_iter_0',
                     branch='left', kernel_size=5)
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_0', name_tf=name_tf + '/comb_iter_0',
                     branch='right', kernel_size=7)

    # comb_iter_1
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_1', name_tf=name_tf + '/comb_iter_1',
                     branch='right', kernel_size=7)

    # comb_iter_2
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_2', name_tf=name_tf + '/comb_iter_2',
                     branch='right', kernel_size=5)

    # comb_iter_4
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_4', name_tf=name_tf + '/comb_iter_4',
                     branch='left', kernel_size=3)


def load_first_cell(state_dict, path, name_pth, name_tf):
    # conv 1x1
    load_conv2d(state_dict, path, name_pth=name_pth + '.conv_1x1.conv', name_tf=name_tf + '/1x1')
    load_bn(state_dict, path, name_pth=name_pth + '.conv_1x1.bn', name_tf=name_tf + '/beginning_bn')

    # other path
    load_conv2d(state_dict, path, name_pth=name_pth + '.path_1.conv', name_tf=name_tf + '/path1_conv')
    load_conv2d(state_dict, path, name_pth=name_pth + '.path_2.conv', name_tf=name_tf + '/path2_conv')
    load_bn(state_dict, path, name_pth=name_pth + '.final_path_bn', name_tf=name_tf + '/final_path_bn')

    # comb_iter_0
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_0', name_tf=name_tf + '/comb_iter_0',
                     branch='left', kernel_size=5)
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_0', name_tf=name_tf + '/comb_iter_0',
                     branch='right', kernel_size=3)

    # comb_iter_1
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_1', name_tf=name_tf + '/comb_iter_1',
                     branch='left', kernel_size=5)
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_1', name_tf=name_tf + '/comb_iter_1',
                     branch='right', kernel_size=3)

    # comb_iter_4
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_4', name_tf=name_tf + '/comb_iter_4',
                     branch='left', kernel_size=3)


def load_normal_cell(state_dict, path, name_pth, name_tf):
    # conv 1x1
    load_conv2d(state_dict, path, name_pth=name_pth + '.conv_1x1.conv', name_tf=name_tf + '/1x1')
    load_bn(state_dict, path, name_pth=name_pth + '.conv_1x1.bn', name_tf=name_tf + '/beginning_bn')

    # conv prev_1x1
    load_conv2d(state_dict, path, name_pth=name_pth + '.conv_prev_1x1.conv', name_tf=name_tf + '/prev_1x1')
    load_bn(state_dict, path, name_pth=name_pth + '.conv_prev_1x1.bn', name_tf=name_tf + '/prev_bn')

    # comb_iter_0
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_0', name_tf=name_tf + '/comb_iter_0',
                     branch='left', kernel_size=5)
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_0', name_tf=name_tf + '/comb_iter_0',
                     branch='right', kernel_size=3)

    # comb_iter_1
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_1', name_tf=name_tf + '/comb_iter_1',
                     branch='left', kernel_size=5)
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_1', name_tf=name_tf + '/comb_iter_1',
                     branch='right', kernel_size=3)

    # comb_iter_4
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_4', name_tf=name_tf + '/comb_iter_4',
                     branch='left', kernel_size=3)


def load_reduction_cell(state_dict, path, name_pth, name_tf):
    # conv 1x1
    load_conv2d(state_dict, path, name_pth=name_pth + '.conv_1x1.conv', name_tf=name_tf + '/1x1')
    load_bn(state_dict, path, name_pth=name_pth + '.conv_1x1.bn', name_tf=name_tf + '/beginning_bn')

    # conv prev_1x1
    load_conv2d(state_dict, path, name_pth=name_pth + '.conv_prev_1x1.conv', name_tf=name_tf + '/prev_1x1')
    load_bn(state_dict, path, name_pth=name_pth + '.conv_prev_1x1.bn', name_tf=name_tf + '/prev_bn')

    # comb_iter_0
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_0', name_tf=name_tf + '/comb_iter_0',
                     branch='left', kernel_size=5)
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_0', name_tf=name_tf + '/comb_iter_0',
                     branch='right', kernel_size=7)

    # comb_iter_1
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_1', name_tf=name_tf + '/comb_iter_1',
                     branch='right', kernel_size=7)

    # comb_iter_2
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_2', name_tf=name_tf + '/comb_iter_2',
                     branch='right', kernel_size=5)

    # comb_iter_4
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_4', name_tf=name_tf + '/comb_iter_4',
                     branch='left', kernel_size=3)


def load(path, num_normal_cells):
    state_dict = {}

    # block1
    load_conv2d(state_dict, path, name_pth='conv0.conv', name_tf='conv0')
    load_bn(state_dict, path, name_pth='conv0.bn', name_tf='conv0_bn')

    # cell_stem
    load_cell_stem_0(state_dict, path, 'cell_stem_0', 'cell_stem_0')
    load_cell_stem_1(state_dict, path, 'cell_stem_1', 'cell_stem_1')
    cell_id = 0
    for i in range(3):
        load_first_cell(state_dict, path, 'cell_'+str(cell_id), 'cell_'+str(cell_id))
        cell_id += 1
        for _ in range(num_normal_cells-1):
            load_normal_cell(state_dict, path, 'cell_' + str(cell_id), 'cell_' + str(cell_id))
            cell_id += 1
        if i < 2:
            load_reduction_cell(state_dict, path, 'reduction_cell_'+str(i), 'reduction_cell_'+str(i))
        else:
            load_linear(state_dict, path, 'linear', 'final_layer/FC')

    return state_dict


def build_and_save_model(path, nas_type):
    path_weights = os.path.join(path, 'weights', nas_type == 'large' and 'NASNet-A_Large_331' or 'NASNet-A_Mobile_224')
    model = NASNetALarge(1001) if nas_type == 'large' else NASNetAMobile(1001)
    state_dict = load(path_weights, model.num_normal_cells)
    model.load_state_dict(state_dict)
    filename_model = os.path.join(path, 'pytorch', nas_type == 'large' and 'nasnet_a_large.pth' or 'nasnet_a_mobile.pth')
    os.system('mkdir -p '+path+'/pytorch')
    torch.save(model.state_dict(), filename_model)
    return model


parser = argparse.ArgumentParser()
parser.add_argument('--nas-type', type=str, choices=['mobile', 'large'], metavar='NASNET_TYPE',
        help='nasnet type: mobile | large')
args = parser.parse_args()

path = './tf-models'
model = build_and_save_model(path, args.nas_type)
model.eval()

print(model)
image_size = args.nas_type == 'mobile' and 224 or 331
input = torch.autograd.Variable(torch.ones(1, 3, image_size, image_size))
output = model.forward(input)
print('output', output)
