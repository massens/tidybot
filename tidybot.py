# (c) Copyright 2017 Marc Assens. All Rights Reserved.

import click
import os
import re


__author__ = "Marc Assens"
__version__ = "0.1"

arg = click.argument
opt = click.option

pwd = os.getcwd()
INPUT_DIR = pwd + '/input/'


@click.group()
def cli():
    """
    Manage the folder structure of your dl experiments
    """
    pass

@cli.command()
@opt('--name', default='', help='Name of the experiment')
def init(name):

		
	# Init	

	# Create folders
	folders = ["eval", "test", "input", "output"]
	for f in folders:
		mkdir(f)

	# Create files
	model(name)

@cli.command()
@opt('--name', default='', help='Name of the experiment')
def model(name):
	"""
	Create a new model
	Generates a 
	"""
	fc = FileCreator()
	name = fc.create_model(name)






# Helper funcitons

def mkdir(folder):
	if not os.path.exists(folder):
		os.mkdir(folder)


class FileCreator:
	file_contents = {"model": "'''\n\tModel: %s\n'''\n\nIN_DIR='%s'\nOUT_DIR ='%s'\n\ndef model():\n\ndef main():\n\nif __name__ == '__main__':\n\tmain()"}

	def __init__(self):
		pass

	def model_text(self, name, in_dir, out_dir):
		return self.file_contents['model'] %(name, in_dir, out_dir) 

	def next_name_for(self, name):
		file_list = os.listdir(pwd)
		files = " ".join(file_list)
		regex = re.compile('%s([0-99]*)' % name)
		indices = regex.findall(files)

		try:
			indices = list(map(int, indices))
			next_i = max(indices) + 1
		except ValueError:
			next_i = 1	

		return name + '%d' % next_i

	def create_model(self, model_name):
		name = self.next_name_for('model')

		if model_name:
			name = name +'_'+ model_name

		print('The next name is %s' % name)

		# Create output dir in /ouput
		model_out_dir= 'output/' + name 
		mkdir(model_out_dir)
		
		# Create model file
		m = open(name +'.py', 'w')
		m.write(self.model_text(name, INPUT_DIR, model_out_dir))
		m.close()
		return name
						

if __name__ == '__main__':
    cli()















    from keras.optimizers import SGD
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.applications.vgg16 import VGG16
    from keras.layers import Input
    from keras.layers.convolutional import Convolution2D, UpSampling2D
    from keras.models import Model
    from keras.losses import binary_crossentropy
    import random
    import glob
    import os
    from datetime import datetime
    from tqdm import tqdm
    import time
    import numpy as np
    import sys
    import cv2
     
    from constants import *
     
     
    def batch_generator(listFilesTrain, BATCH_SIZE):
     
        while True:
            random.shuffle(listFilesTrain)
            for idx in range(0, len(listFilesTrain), BATCH_SIZE):
                ids = slice(idx, idx + BATCH_SIZE)
                images = np.asarray([cv2.cvtColor(cv2.imread(os.path.join(pathOutputImages, name + '.png'), cv2.IMREAD_COLOR),
                                                  cv2.COLOR_BGR2RGB) # .transpose(2, 0, 1)
                                     for name in listFilesTrain[ids]], dtype='float32')
                targets = np.expand_dims(np.asarray([cv2.cvtColor(cv2.imread(os.path.join(pathOutputMaps, name + '.png'),
                                                                             cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY).astype('float32') / 255.
                                                     for name in listFilesTrain[ids]], dtype='float32'), axis=3)
                # print images.shape, targets.shape
                yield (images, targets)
     
     
    def model():
     
        input_tensor = Input([params['image_height'], params['image_width'], 3])
     
        encoder = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
        conv5_3 = encoder.get_layer('block5_conv3').output
     
        # Deconder unpooling + convolution
        uconv5_1 = Convolution2D(512, 3, activation='relu', border_mode='same')(conv5_3)
        uconv5_2 = Convolution2D(512, 3, activation='relu', border_mode='same')(uconv5_1)
        uconv5_3 = Convolution2D(512, 3, activation='relu', border_mode='same')(uconv5_2)
     
        uconv4_pool = UpSampling2D((2, 2))(uconv5_3)
        uconv4_1 = Convolution2D(512, 3, activation='relu', border_mode='same')(uconv4_pool)
        uconv4_2 = Convolution2D(512, 3, activation='relu', border_mode='same')(uconv4_1)
        uconv4_3 = Convolution2D(512, 3, activation='relu', border_mode='same')(uconv4_2)
     
        uconv3_pool = UpSampling2D((2, 2))(uconv4_3)
        uconv3_1 = Convolution2D(256, 3, activation='relu', border_mode='same')(uconv3_pool)
        uconv3_2 = Convolution2D(256, 3, activation='relu', border_mode='same')(uconv3_1)
        uconv3_3 = Convolution2D(256, 3, activation='relu', border_mode='same')(uconv3_2)
     
        uconv2_pool = UpSampling2D((2, 2))(uconv3_3)
        uconv2_1 = Convolution2D(128, 3, activation='relu', border_mode='same')(uconv2_pool)
        uconv2_2 = Convolution2D(128, 3, activation='relu', border_mode='same')(uconv2_1)
     
        uconv1_pool = UpSampling2D((2, 2))(uconv2_2)
        uconv1_1 = Convolution2D(64, 3, activation='relu', border_mode='same')(uconv1_pool)
        uconv1_2 = Convolution2D(64, 3, activation='relu', border_mode='same')(uconv1_1)
     
        output = Convolution2D(1, 1, activation='sigmoid', border_mode='same')(uconv1_2)
     
        model = Model(input=[input_tensor], output=[output])
     
        for layer in model.layers:
            print(layer.input_shape, layer.output_shape)
     
        return model
     
     
    num_epochs = 30
     
     
    def main():
        """
       Traininig with single gpu
       :return:
       """
        list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathOutputMaps, '*'))]
        list_files_train = [k for k in list_img_files if 'train' in k]
        list_files_val = [k for k in list_img_files if 'val' in k]
     
        train_generator = batch_generator(list_files_train, BATCH_SIZE)
        val_generator = batch_generator(list_files_val, BATCH_SIZE)
     
        generator = model()
     
        sgd = SGD(lr=1e-3, decay=0.005, momentum=0.9, nesterov=True)
        print("Compile Model")
        generator.compile(sgd, 'binary_crossentropy')
     
        print("Training Model")
        generator.fit_generator(generator=train_generator, steps_per_epoch=int(10000/BATCH_SIZE), nb_epoch=90,
                                validation_data=val_generator, validation_steps=1, workers=1,
                                callbacks=[EarlyStopping(patience=5),
                                           ModelCheckpoint('weights.generator.{epoch:02d}-{val_loss:.4f}.pkl',
                                                           save_best_only=True)])
     
     
    if __name__ == '__main__':
        main()