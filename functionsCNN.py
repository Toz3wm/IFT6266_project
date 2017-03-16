import os, sys
import glob
import cPickle as pkl
import numpy as np
import PIL.Image as Image
from skimage.transform import resize
import time
import pickle

import theano
import theano.tensor as T
from theano import shared
import random

import lasagne


def scale(A, B, k):     
    # fill A with B scaled by k
    Y = A.shape[0]
    X = A.shape[1]
    for y in range(0, k):
        for x in range(0, k):
            A[y:Y:k, x:X:k] = B



def load_database(size, data_dir_input = 'inpainting/train2014cropped', data_dir_target = 'inpainting/train2014target'):
    """
    creates a list of batch_size pictures, randomly chosen
    batch_size should be set to train set size, but for testing reason is default = 5
    """
    DataBaseInput = np.zeros((size,64,64,3),dtype = 'float32')
    DataBaseTarget = np.zeros((size,32,32,3),dtype= 'float32')
    imgsInput = glob.glob(data_dir_input+"/*.jpg")
    imgsTarget = glob.glob(data_dir_target+"/*.jpg")
    if not len(imgsInput) == len(imgsTarget):
        raise Exception("inputs and target are not the same size !")
    for i in range(size):
        img_path_input = imgsInput[i] 
        img_input = Image.open(img_path_input)
        img_input = np.asarray(img_input, dtype='float32') / 256.

        if  len(img_input.shape) == 3:
            DataBaseInput[i,:,:,:] = img_input[:,:,:]
            img_path_target = imgsTarget[i]
            img_target = Image.open(img_path_target)
            img_target = np.asarray(img_target, dtype='float32') / 256.
            #if not np.ma.shape(img_target) == (32,32):
            #    raise Exception("Wrong shape :"+str(np.ma.shape(img_target)))
            #img_target = np.reshape(img_target, 1024)
            DataBaseTarget[i,:,:,:] = img_target[:,:,:]

    DataBaseInput = np.transpose(DataBaseInput,(0,3,1,2))
    DataBaseTarget = np.transpose(DataBaseTarget,(0,3,1,2))
    return DataBaseInput, DataBaseTarget

def select_random_batch(batch_size, DataBaseInput, DataBaseTarget):

    indices = range(len(DataBaseInput))
    random.shuffle(indices)
    selected_indices = indices[0:batch_size]
    #print(selected_indices)

    return DataBaseInput[selected_indices,:,:,:], DataBaseTarget[selected_indices,:,:,:]


def build_cnn(input_var=None):
    

    #Input layer
    input_layer = lasagne.layers.InputLayer(
            shape=(None, 3, 64, 64),input_var=input_var)
    print(input_layer.output_shape)
    network =  input_layer

    #conv layer, with 32 5x5 sized kernels, using ReLu
    conv1_layer = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    print(conv1_layer.output_shape)
    network = conv1_layer

    #max pooling layer
    pool1_layer = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    print(pool1_layer.output_shape)
    network = pool1_layer

    #second conv layer, still with 32 5x5 sized kernels, using ReLu
    conv2_layer = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    print(conv2_layer.output_shape)
    network = conv2_layer

    #second max pooling layer :
    pool2_layer = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    print(pool2_layer.output_shape)
    network = pool2_layer

    #fully connected layer, with dropout 
    dense1_layer = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify)
    print(dense1_layer.output_shape)
    network = dense1_layer

    #fully connected layer, with dropout 
    dense2_layer = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify)
    print(dense2_layer.output_shape)
    network = dense2_layer


    #reshaping the 512 features representation
    reshape_layer = lasagne.layers.ReshapeLayer(network, ((input_var.shape[0],32,4,4)))
    print(reshape_layer.output_shape)
    network = reshape_layer

    # transpose conv layer to reconstruct
    deconv1_layer = lasagne.layers.TransposedConv2DLayer(network, 16, (3, 3), stride=(1, 1))
    print(deconv1_layer.output_shape)
    network = deconv1_layer

    #upscaling to 3x16x16
    upscale1_layer = lasagne.layers.Upscale2DLayer(network, (2,2))
    print(upscale1_layer.output_shape)
    network = upscale1_layer

    # transpose conv layer to reconstruct, with sigmoid output
    deconv2_layer = lasagne.layers.TransposedConv2DLayer(network, 8, (3,3), stride=(1, 1))
    print(deconv2_layer.output_shape)
    network = deconv2_layer


    #upscaling to 3x16x16
    upscale1_layer2 = lasagne.layers.Upscale2DLayer(network, (2,2))
    print(upscale1_layer2.output_shape)
    network = upscale1_layer2

    # transpose conv layer to reconstruct, with sigmoid output
    deconv3_layer = lasagne.layers.TransposedConv2DLayer(network, 3, (5, 5), stride=(1, 1),nonlinearity=lasagne.nonlinearities.sigmoid)
    print(deconv3_layer.output_shape)
    network = deconv3_layer

    return network


def network_loss(predicted, real):

    print("shape of real"+ str(real.shape.eval()))
    #print("shape of predicted"+ str(predicted.shape.eval()))

    predictedReshape = predicted
    #predictedReshape = np.zeros((32,32),dtype = predicted.dtype)
    #scale(predictedReshape,predicted, 2)
    loss = -(predicted - real)
    loss = loss.mean(axis = 0)
    return loss


def main():
    num_epochs = 1000000
    batch_size = 200
    val_batch_size = 20000
    database_size = 80000
    basicFilename ="basicConvolutionnalModel"

    print("Loading training database")
    DatabaseInput, DatabaseTarget = load_database(database_size,'inpainting/train2014cropped','inpainting/train2014target')
    print("Loading validation database")
    DatabaseValInput, DatabaseValTarget = load_database(database_size,'inpainting/val2014cropped','inpainting/val2014target')
    input_var = T.tensor4('inputs')

    batch_input, batch_target = select_random_batch(batch_size, DatabaseInput, DatabaseTarget)

    print("building the model ...")
    network = build_cnn(input_var)

    prediction = lasagne.layers.get_output(network)
   # print("output of the network :")
   # print(lasagne.utils.as_theano_expression(prediction).get_shape())
    get_network_output = theano.function([input_var], [prediction],allow_input_downcast=True)

    batchTarget_shared = shared(batch_target)
    loss = T.mean(lasagne.objectives.squared_error(prediction,batchTarget_shared))
    #loss = network_loss(prediction, DataBaseTarget_shared)

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adagrad(loss, params, learning_rate=0.001)

    train_fn = theano.function([input_var], [loss], updates=updates)
    validation_fn = theano.function([input_var],[loss])

    print("Start training !")

    validation_error = 1
    val_and_save_iteration = 50

    for epoch in range(num_epochs):

        start_time = time.time()

        batch_input, batch_target = select_random_batch(batch_size, DatabaseInput, DatabaseTarget)

        #print("Batch created, of shape" + str(np.shape(batch_input)) +"for input  and "+ str(np.shape(batch_target))+"for target")
        [predictions] = get_network_output(batch_input)

        batchTarget_shared.set_value(batch_target)
        [train_err] = train_fn(batch_input)

        print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err /1))
        if epoch%val_and_save_iteration == 0:

            print("validation ...")
            batch_input, batch_target = select_random_batch(val_batch_size, DatabaseValInput, DatabaseValTarget)
            [predictions] = get_network_output(batch_input)
            batchTarget_shared.set_value(batch_target)
            [val_err] = validation_fn(batch_target)
            saveModel(network, basicFilename + "_ite_" + str(epoch + 1) + "_bs_" + str(batch_size) + "_val_err_" + str(val_err))
            print("  validation loss:\t\t{:.6f}".format(val_err /1))
            
            if validation_error < val_err:
                validation_error = val_err
            else:
                print "Validation error growing... Stoping training"
                break



        # if epoch % 100 == 0:
        #    np.savez('model.npz', *lasagne.layers.get_all_param_values(network))

    print("Training done !")


def saveModel(network, filename):
    values = lasagne.layers.get_all_param_values(network)
    with open(filename+"pickle", 'wb') as handle:
        pickle.dump(values,handle,protocol=pickle.HIGHEST_PROTOCOL)
    i = 0


def loadModel(network, filename):
    values = None
    with open(filename+"pickle", 'rb') as handle:
        values = pickle.load(handle)
    lasagne.layers.set_all_param_values(network, values)
    i = 0
    
main()