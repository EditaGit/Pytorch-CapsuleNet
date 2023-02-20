import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt
from numpy import genfromtxt
from capsnet import CapsNet
from data_loader import Dataset
from tqdm import tqdm
from PIL import Image, ImageOps
import io

USE_CUDA = True if torch.cuda.is_available() else False
BATCH_SIZE = 10
N_EPOCHS = 25
LEARNING_RATE = 0.005
MOMENTUM = 0.9

'''
Config class to determine the parameters for capsule net
'''


class Config:
    def __init__(self, dataset='mnist'):
        if dataset == 'mnist':
            # CNN (cnn)
            self.cnn_in_channels = 1
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 6 * 6

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = 32 * 6 * 6
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 28
            self.input_height = 28

        elif dataset == 'cifar10':
            # CNN (cnn)
            self.cnn_in_channels = 3
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 8 * 8

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = 32 * 8 * 8
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 32
            self.input_height = 32

        elif dataset == 'your own dataset':
            self.cnn_in_channels = 1
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 6 * 6

            # Digit Capsule (dc)
            self.dc_num_capsules = 2
            self.dc_num_routes = 32 * 6 * 6
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 28
            self.input_height = 28


def train(model, optimizer, train_loader, epoch):
    torch.zeros(1).cuda()

    capsule_net = model
    capsule_net.train()
    n_batch = len(list(enumerate(train_loader)))
    total_loss = 0
    for batch_id, (data, target) in enumerate(tqdm(train_loader)):

        target = torch.sparse.torch.eye(2).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)
        loss.backward()
        optimizer.step()
        correct = sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1))
        train_loss = loss.item()
        total_loss += train_loss
        if batch_id % 100 == 0:
            tqdm.write("Epoch: [{}/{}], Batch: [{}/{}], train accuracy: {:.6f}, loss: {:.6f}".format(
                epoch,
                N_EPOCHS,
                batch_id + 1,
                n_batch,
                correct / float(BATCH_SIZE),
                train_loss / float(BATCH_SIZE)
            ))

    tqdm.write('Epoch: [{}/{}], train loss: {:.6f}'.format(epoch, N_EPOCHS, total_loss / len(train_loader.dataset)))


def test(capsule_net, test_loader, epoch=None):
    capsule_net.eval()
    test_loss = 0
    correct = 0
    for batch_id, (data, target) in enumerate(test_loader):
        target = torch.sparse.torch.eye(2).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)

        test_loss += loss.item()
        correct += sum(np.argmax(masked.data.cpu().numpy(), 1) ==
                       np.argmax(target.data.cpu().numpy(), 1))

    tqdm.write(
        "Epoch: [{}/{}], test accuracy: {:.6f}, loss: {:.6f}".format(epoch, N_EPOCHS,
                                                                     correct / len(test_loader.dataset),
                                                                     test_loss / len(test_loader)))


def display_images(images, reconstructions):
    '''Plot one row of original MNIST images and another row (below)
       of their reconstructions.'''
    # convert to numpy images
    images = images.data.cpu().numpy()
    reconstructions = reconstructions.view(-1, 1, 28, 28)
    reconstructions = reconstructions.data.cpu().numpy()

    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(26, 5))

    # input images on top row, reconstructions on bottom
    for images, row in zip([images, reconstructions], axes):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()
def tester(capsule_net, test_loader):
    capsule_net.eval()

    correct = 0
    for batch_id, (data, target) in enumerate(test_loader):

        target = torch.sparse.torch.eye(2).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        #plt.imshow(T.ToPILImage()(data[0]))
        #plt.show()

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        # print(batch_id, data.shape)
        output, reconstructions, masked = capsule_net(data)
        print(masked)
        print()
        print()
        print()
        print()

        for i, img in enumerate(data):
            output, reconstructions, masked_ = capsule_net(img.unsqueeze(0))
            print(masked_)
        #     print(img.shape)
        #     # my_dataset = TensorDataset(img[None,:,:,:])
        #     # output, reconstructions, masked = capsule_net(DataLoader(my_dataset))
        #     output, reconstructions, masked = capsule_net(img[None,:,:,:].float())
        #
        #     print(masked)
        #     # classify_one_sample(capsule_net, torch.unsqueeze(img, 0))

        #plt.imshow(T.ToPILImage()(reconstructions[0]))
        #plt.show()

        correct += sum(np.argmax(masked.data.cpu().numpy(), 1) ==
                       np.argmax(target.data.cpu().numpy(), 1))

        #display_images(data,reconstructions)

    print("Test_accuracy:",correct / len(test_loader.dataset))

def classify_one_sample(capsule_net, bb):
    capsule_net.eval()
    data = Variable(bb)

    #print("Data nacitane v sample")
    #plt.imshow(T.ToPILImage()(data[0]))
    #plt.show()

    if USE_CUDA:
        data = data.cuda()

    output, reconstructions, masked = capsule_net(data)

    #display_images(data,reconstructions)


    #print("Rekonstrukcia")
    #plt.imshow(T.ToPILImage()(reconstructions[0]))
    #plt.show()
    return np.argmax(masked.data.cpu().numpy())


def transform_image(img):
    transform = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Grayscale()
    ])

    return transform(img).to('cuda').unsqueeze(0)

def split_fingerprint_and_classify_bb(capsule_net, test_sample, txt):
    data_to_txt =[]
    #img = ImageOps.grayscale(Image.open(test_sample))
    my_data = genfromtxt(txt, delimiter=' ')

    for line in my_data:
        img = Image.open(test_sample)
        dw, dh = img.size
        feature, x, y, w, h = line[0],line[1], line[2], line[3], line[4]

        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)

        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1

        img = img.crop((l, t, r, b))
        #img_res = img_res.resize((28, 28))
        #img_res = np.expand_dims(np.expand_dims(np.array(img_res).astype('float32'), axis=0), axis=0)/255
        #capsnet_feature = classify_one_sample(capsule_net, torch.from_numpy(img_res))
        #print("Povodny obrazok")
        #plt.imshow(img)
        #plt.show()

        capsnet_feature = classify_one_sample(capsule_net, transform_image(img))

        one_bb = [int(feature), x, y, w, h, capsnet_feature]

        print(one_bb)
        data_to_txt.append(one_bb)
    np.savetxt('vystup.txt', data_to_txt)



if __name__ == '__main__':
    torch.manual_seed(1)

    dataset = 'your own dataset'

    config = Config(dataset)
    mnist = Dataset(dataset, BATCH_SIZE)
    #
    #capsule_net = CapsNet(config)
    #capsule_net = torch.nn.DataParallel(capsule_net)
    #if USE_CUDA:
    #   capsule_net = capsule_net.cuda()
    #capsule_net = capsule_net.module
#
    #optimizer = torch.optim.Adam(capsule_net.parameters())
#
    #for e in range(0, N_EPOCHS + 0):
    #  train(capsule_net, optimizer, mnist.train_loader, e)
    #  test(capsule_net, mnist.test_loader, e)
    #  os.makedirs('model_delete_softmax', exist_ok=True)
    #  torch.save(capsule_net, 'model_delete_softmax/checkpoint_epoch' + str(e))

    model = torch.load('model_valid/checkpoint_epoch49')

    #model = torch.load('model_delete_softmax/checkpoint_epoch24')


    #split_fingerprint_and_classify_bb(model,
    #                                  "/home/edka/PycharmProjects/Pytorch-CapsuleNet/images/101_1.tif",
    #                                  "/home/edka/PycharmProjects/Pytorch-CapsuleNet/images/101_1.txt")

    #my_data = genfromtxt("103_3_vystup.txt", delimiter=' ')
    #print("iauhdpiouhdpwuu")
    #print(float(my_data[0][1]))


    split_fingerprint_and_classify_bb(model, "/home/edka/PycharmProjects/Pytorch-CapsuleNet/prezentacia/prezentacia_labelImg/101_3.tif","/home/edka/PycharmProjects/Pytorch-CapsuleNet/prezentacia/prezentacia_labelImg/101_3.txt")

    #tester(model, mnist.same)


