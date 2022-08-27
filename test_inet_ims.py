import random
import tarfile

random.seed(3942)

n = 1100 # number of files to get

file = '/home/trevor/Downloads/imagenet_object_localization_patched2019.tar.gz'
#file = '/home/trevor/Downloads/dumb_ims.tar.gz'
tarf = tarfile.open(file)
all_files = tarf.getnames()
test_files = [fi for fi in all_files if fi.startswith('ILSVRC/Data/CLS-LOC/test/')]
#test_files = [fi for fi in all_files if fi.startswith('dumb_ims/aaa/')]
e_files = random.sample(test_files, n) 
for i in range(n):
    tarf.extract(e_files[i], path='/home/trevor/Downloads')
tarf.close()

