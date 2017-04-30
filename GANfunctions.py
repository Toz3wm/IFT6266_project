import os, sys
import glob
import cPickle as pkl
import numpy as np
import PIL.Image as Image
from skimage.transform import resize
import time
import pickle
import examples

import theano
import theano.tensor as T
from theano import function
from theano import shared
import random

import lasagne
import functionsCNN


# will generate samples from noise
def generator(input_var=None):
    print "building generator"
    num_units = 512  # should be a multiple of 16
    noise_size = 100  # size of noise vector

    network = lasagne.layers.InputLayer(shape=(None, noise_size), input_var=input_var)
    print(network.output_shape)
    # setting the shape to a 4D tensor
    network = lasagne.layers.ReshapeLayer(network, (-1, noise_size, 1, 1))
    print(network.output_shape)

    # first layer, num_units filters of size 4x4
    network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network, num_units, (4, 4), stride=(1, 1)))
    print(network.output_shape)

    # second layer, num_units/2 filters of size 8x8
    network = lasagne.layers.batch_norm(
        lasagne.layers.TransposedConv2DLayer(network, num_units / 2, (5, 5), stride=(2, 2), crop=2, output_size=8))
    print(network.output_shape)

    # third layer, num_units/4 filters of size 16x16
    network = lasagne.layers.batch_norm(
        lasagne.layers.TransposedConv2DLayer(network, num_units / 4, (5, 5), stride=(2, 2), crop=2, output_size=16))
    print(network.output_shape)

    # last layer, 3 filters (R,G,B) of size 32x32 as expected
    network = lasagne.layers.TransposedConv2DLayer(network, num_filters=3, filter_size=(5, 5), stride=(2, 2), crop=2,
                                                   nonlinearity=lasagne.nonlinearities.sigmoid, output_size=32)
    print(network.output_shape)

    return network


# learn to separate artificial pictures from real ones
def discriminator(input_var=None):
    print "building discriminator"
    num_units = 100
    lrelu = lasagne.nonlinearities.LeakyRectify(0.2)

    # the input is a picture of size 3x32x32
    network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=input_var)
    print(network.output_shape)

    # three convolutional layers to encode the picture
    network = lasagne.layers.Conv2DLayer(network, num_filters=num_units / 4, filter_size=(5, 5), stride=2, pad=2,
                                         nonlinearity=lrelu)
    print(network.output_shape)

    network = lasagne.layers.batch_norm(
        lasagne.layers.Conv2DLayer(network, num_filters=num_units / 2, filter_size=(5, 5), stride=2, pad=2,
                                   nonlinearity=lrelu))
    print(network.output_shape)

    network = lasagne.layers.batch_norm(
        lasagne.layers.Conv2DLayer(network, num_filters=num_units, filter_size=(5, 5), stride=2, pad=2,
                                   nonlinearity=lrelu))
    print(network.output_shape)

    # we plug a denselayer with only one neuron as the output. This neuron performs binary classification with a sigmoid function.
    network = lasagne.layers.FlattenLayer(network)
    print(network.output_shape)
    network = lasagne.layers.DenseLayer(network, 1,
                                        nonlinearity=lasagne.nonlinearities.sigmoid)
    print(network.output_shape)
    return network


def main():
    num_epochs = 1000000
    database_size = 30000
    startpoint = 1
    validation_error = 1
    val_iteration = 300
    save_iteration = 600
    batch_size = 128

    basicFilename = "GAN_Basic_"
    loading = False
    filename = "models/CNN_ReLu__nbPic_10500000_val_err_0.00018pickle"

    print("Loading training database")
    DatabaseInput, DatabaseTarget = functionsCNN.load_database(database_size, startpoint, 'inpainting/train2014cropped',
                                                               'inpainting/train2014target')

    input_var = T.tensor4('inputsDisc')
    input_var_gen = T.matrix('inputsGen')
    batch_input, batch_target = functionsCNN.select_random_batch(batch_size, DatabaseInput, DatabaseTarget)

    print("building the model ...")

    generatorN = generator(input_var_gen)
    discriminatorN = discriminator(input_var)

    batch_target_shared = shared(batch_target)

    generated_samples = lasagne.layers.get_output(generatorN)
    generated_score = lasagne.layers.get_output(discriminatorN, generated_samples)
    realsample_score = lasagne.layers.get_output(discriminatorN, batch_target_shared)

    # print("output of the network :")
    # print(lasagne.utils.as_theano_expression(prediction).get_shape())

    get_gen_output = theano.function([input_var_gen], [generated_samples], allow_input_downcast=True)

    get_disc_output_real = theano.function([input_var], [realsample_score], allow_input_downcast=True,
                                           on_unused_input='warn')
    get_disc_output_gen = theano.function([input_var, input_var_gen], [generated_score], allow_input_downcast=True,
                                          on_unused_input='warn')

    genloss, discloss = gan_non_saturating_loss(generated_score, realsample_score)

    genparams = lasagne.layers.get_all_params(generatorN, trainable=True)
    genupdates = lasagne.updates.adam(genloss, genparams, learning_rate=0.0002, beta1 = 0.5)

    discparams = lasagne.layers.get_all_params(discriminatorN, trainable=True)
    discupdates = lasagne.updates.adam(discloss, discparams, learning_rate=0.0002, beta1 = 0.5)

    train_generator = theano.function([input_var_gen], [genloss], updates=genupdates, allow_input_downcast=True)
    train_discriminator = theano.function([input_var, input_var_gen], [discloss], updates=discupdates,
                                          on_unused_input='warn', allow_input_downcast=True)

    if loading:
        loadModel(network, filename)

    print("Start training !")

    gen_error_list = ""
    disc_error_list = ""

    for epoch in range(num_epochs):

        start_time = time.time()

        batch_input, batch_target = functionsCNN.select_random_batch(batch_size, DatabaseInput, DatabaseTarget)

        noise = np.random.normal(size=(batch_size, 100))
        [generatedSample] = get_gen_output(noise)

        batch_target_shared.set_value(batch_target)

        [gen_err] = train_generator(noise)
        [disc_err] = train_discriminator( generatedSample, noise)

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))

        print("	         generator loss:\t\t{:.6f}".format(gen_err / 1))
        print("	     discriminator loss:\t\t{:.6f}".format(disc_err / 1))

        gen_error_list += "{}, {:.8f} \n".format(epoch+1, gen_err/1)
	disc_error_list += "{}, {:.8f} \n".format(epoch+1, disc_err/1)

        if epoch % val_iteration == 0:
            if epoch % save_iteration == 0:
                print("saving model")
                functionsCNN.saveModel(generatorN, "models/" + basicFilename + "_nbPic_{}GEN".format(epoch * batch_size))
                functionsCNN.saveModel(discriminatorN, "models/" + basicFilename + "_nbPic_{}GEN".format(epoch * batch_size))
            with open("models/" + basicFilename + 'gen_err.txt', 'a') as file:
                file.write(gen_error_list)
            with open("models/" + basicFilename + 'disc_err.txt', 'a') as file:
                file.write(disc_error_list)
            gen_error_list = ""
            disc_error_list = ""

            print "creating visual examples from validation dataset"
            for i in range(0, 2):
                res = np.transpose(
                    (generatedSample[i, :, :, :] * 256 -1).astype(
                        'uint8'), (1, 2, 0))

                imgRes = Image.fromarray(res)

                imgRes.save("inpainting/visualExamplesGAN/" + basicFilename + str(i) + "_nbPic_{}_Res.jpg".format(
                    epoch * batch_size))



    print("Training done !")


def gan_likelihood_loss(generated_score, realsample_score):
    discloss = -0.5 * (T.log(realsample_score) + T.log(1 - generated_score)).mean()
    genloss = -0.5 * (generated_score / (1 - generated_score)).mean()

    return genloss, discloss


def gan_non_saturating_loss(generated_score, realsample_score):
    discloss = -0.5 * (T.log(realsample_score) + T.log(1 - generated_score)).mean()
    genloss = -0.5 * T.log(generated_score).mean()

    return genloss, discloss
	




if __name__ == '__main__':
    main()
