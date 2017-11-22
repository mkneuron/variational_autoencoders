import variational_autoencoders as autoencoders
import numpy as np
import matplotlib.pyplot as plt
from chainer import iterators, optimizers, serializers
from chainer import functions as F
from mnist import MNIST

# load data
mndata = MNIST('/home/mknull/python-mnist/data')
mndata.load_training()
mndata.load_testing()

# define layer size variables
latent_size = 10 # length of latent vector
n_monte_carlo = 10 # number of sampling from the latent layer
hidden = 100
image_l = 784  # length of image

max_epoch = 10
batch_size = 100

use_noise = False

# init iterators
train_iterator = iterators.SerialIterator(mndata.train_images, batch_size=batch_size)
test_iterator = iterators.SerialIterator(mndata.test_images, batch_size=batch_size, repeat=False, shuffle=True)

# define networks
# Encoder = autoencoders.ConvolutionalEncoder(out=latent_l, hidden=100)
# Decoder = autoencoders.DeconvolutionalDecoder(out=image_l, _in=latent_l, hidden=784, batch_size=batch)

model = autoencoders.AutoEncoder(latent_size=latent_size, input_size=image_l,batch_size=batch_size)

# setup optimizers
# optimizer1 = optimizers.adam.Adam()
# optimizer2 = optimizers.adam.Adam()
# optimizer1.setup(Encoder)
# optimizer2.setup(Decoder)
optimizer = optimizers.adam.Adam()
optimizer.setup(model)
while train_iterator.epoch < max_epoch:

    # train once
    mn_digits = train_iterator.next()
    mn_digits = np.array(np.divide(mn_digits, 255) - 0.5)
    mn_digits = np.reshape(mn_digits,[batch_size, 28, 28]).astype('float32')

    # # latent = Encoder(mn_digits)
    # # our_digits = Decoder(latent)
    # # # our_digits = Decoder(Encoder(mn_digits))
    # our_digits, latent = model(mn_digits, return_latent=True, add_noise=False)
    #
    # loss_decoder = F.mean_absolute_error(mn_digits, our_digits)
    # # latent_means = F.mean(latent[:, :len(latent.data[1])//2],axis=0)
    # # latent_stds = F.mean(latent[:, len(latent.data[1])//2:], axis=0)
    #
    # #     latent_means = latent[0][:len(latent.data[0])//2]
    # #     latent_stds = latent[0][len(latent.data[0])//2:]
    # #     latent_stds.data = np.log(latent_stds.data)
    # #     loss_encoder = F.gaussian_kl_divergence(latent_means,latent_stds)

    means, variances = model.encode(mn_digits)
    loss_encoder = 0
    loss_decoder = 0

    for i in range(n_monte_carlo):
        if not use_noise:
            z = F.gaussian(means, variances)
        else:
            noise = np.random.random_sample(means.shape).astype('float32')
            means.data = means.data + noise
            z = F.gaussian(means, variances)
            means.data = means.data - noise

        generated_digits = F.squeeze(model.decode(z))
        loss_decoder += F.mean_squared_error(mn_digits,generated_digits)/n_monte_carlo

    variances = F.log(variances)
    loss_encoder += F.gaussian_kl_divergence(means, variances)
    loss = loss_encoder + loss_decoder
    # Encoder.cleargrads()
    # Decoder.cleargrads()
    model.cleargrads()
    loss = loss_decoder + loss_encoder
    loss.backward()
    optimizer.update()
    # optimizer1.update()
    # optimizer2.update()
    print('loss: '+str(loss.data))
    if train_iterator.is_new_epoch:
        print('epoch:{:02d} train_loss:{:.04f} '.format(train_iterator.epoch, float(loss.data)), end='')

        plt.ion()
        data = generated_digits.data

        image0 = np.reshape(data[0], [28, 28])
        image1 = np.reshape(data[1], [28, 28])
        image2 = np.reshape(data[2], [28, 28])
        image3 = np.reshape(data[3], [28, 28])

        fig = plt.figure()
        plt.subplot(221)
        plt.imshow(image0, cmap='gray_r')
        plt.subplot(222)
        plt.imshow(image1, cmap='gray_r')
        plt.subplot(223)
        plt.imshow(image2, cmap='gray_r')
        plt.subplot(224)
        plt.imshow(image3, cmap='gray_r')

        plt.savefig('image_sample_epoch' + str(train_iterator.epoch) + '.png')

# serializers.save_npz('decoder.model', Decoder)
# serializers.save_npz('encoder.model', Encoder)