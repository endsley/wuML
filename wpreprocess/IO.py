#!/usr/bin/env python

import os 
from pathlib import Path

def read_txt_file(path):
	f = open(path, 'r')
	content = f.read()
	f.close()
	return content

def pickle_dump(obj, path):
	pickle.dump( obj, open( path, "wb" ) )
   
def pickle_load(pth):
	return pickle.load( open( pth, "rb" ) )

def copy_file(src, destination):
	copyfile(src, destination)

def write_to(txt, path):
	f = open(path, 'w')
	f.write(txt)
	f.close()

def go_up_1_directory(full_path):	# path should be a string
	return str(Path(full_path).parents[0])

def initialize_empty_folder(path):
	ensure_path_exists(path)
	remove_files(path)

def file_exists(path):
	if os.path.exists(path): return True
	return False

def ensure_path_exists(path):
	if os.path.exists(path): return True
	os.mkdir(path)
	return False

def path_list_exists(path_list):
	for i in path_list:
		if os.path.exists(i) == False: 
			return False

	return True

def create_file(path):
	fin = open(path,'w')
	fin.close()

def delete_file(path):
	if os.path.exists(path):
		os.remove(path)

def remove_files(folder_path):	# make sure to end the path with /
	if not os.path.exists(folder_path): return 

	file_in_tmp = os.listdir(folder_path)
	for i in file_in_tmp:
		if os.path.isfile(folder_path + i):
			os.remove(folder_path + i)

