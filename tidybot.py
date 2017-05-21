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
		model_out_dir= 'output/' + name +'/'
		mkdir(model_out_dir)
		
		# Create model file
		m = open(name +'.py', 'w')
		m.write(self.model_text(name, INPUT_DIR, model_out_dir))
		m.close()
		return name
						

if __name__ == '__main__':
    cli()