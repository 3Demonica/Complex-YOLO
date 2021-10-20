import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np

from complexYOLO import ComplexYOLO
from kitti import KittiDataset
from region_loss import RegionLoss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training complex yolo.')
    parser.add_argument('--dir', default='/home/m_vogel/KITTI', help='directory of KITTI')
    parser.add_argument('--batch_size', default=12, help='batch size')
    parser.add_argument('--epochs', default=200, help='number of epochs')

    args = parser.parse_args()

    dir = args.dir
    batch_size = args.batch_size
    number_epochs = args.epochs

    # dataset
    dataset = KittiDataset(root=dir, set='train')
    data_loader = data.DataLoader(dataset, batch_size, shuffle=True)

    model = ComplexYOLO()
    model.cuda()

    # define optimizer
    optimizer = optim.Adam(model.parameters())

    # define loss function
    region_loss = RegionLoss(num_classes=8, num_anchors=5)

    for epoch in range(number_epochs):

        for batch_idx, (rgb_map, target) in enumerate(data_loader):
            optimizer.zero_grad()

            rgb_map = rgb_map.view(rgb_map.data.size(0), rgb_map.data.size(3), rgb_map.data.size(1),
                                   rgb_map.data.size(2))
            output = model(rgb_map.float().cuda())

            loss = region_loss(output, target)
            loss.backward()
            optimizer.step()

        if (epoch % 10 == 0):
            torch.save(model, "ComplexYOLO_epoch" + str(epoch))
