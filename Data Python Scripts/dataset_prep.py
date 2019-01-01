# Arjun Subramonian

import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

import os
import re

'''Rename folders in Stanford Dogs Dataset.'''
rootdir = '../Stanford_Dogs_Dataset/'

for subdir, dirs, files in os.walk(rootdir):
    try:
        last_digit = re.match('.+([0-9])[^0-9]*$', str(subdir))
        breed = str(subdir)[last_digit.start(1) + 2:].replace('-', '_')

        src = str(subdir)
        dst = str(subdir).rsplit('/', 1)[0] + '/' + breed

        os.rename(src, dst)
    except:
        continue

num_px = 128

breeds = []

for subdir, dirs, files in os.walk(rootdir):
    breeds.append(str(subdir).rsplit('/', 1)[1])

    '''Preprocess images (rescale images, maintaining aspect ratio, and center crop).'''
    for file in files:
        try:
            fname = os.path.join(subdir, file)
            img = Image.open(fname)

            # resize: (width, height)
            # crop: (left, upper, right, lower)
            if img.size[0] < img.size[1]:
                basewidth = num_px
                wpercent = (basewidth / float(img.size[0]))
                hsize = int((float(img.size[1]) * float(wpercent)))
                img = img.resize((basewidth, hsize), Image.ANTIALIAS)

                width, height = img.size
                new_width, new_height = num_px, num_px

                left = (width - new_width)/2
                top = (height - new_height)/2
                right = (width + new_width)/2
                bottom = (height + new_height)/2

                img = img.crop((left, top, right, bottom))

                img.save(fname)
            else:
                sideheight = num_px
                hpercent = (sideheight / float(img.size[1]))
                wsize = int((float(img.size[0]) * float(hpercent)))
                img = img.resize((wsize, sideheight), Image.ANTIALIAS)

                width, height = img.size
                new_width, new_height = num_px, num_px

                left = (width - new_width)/2
                top = (height - new_height)/2
                right = (width + new_width)/2
                bottom = (height + new_height)/2

                img = img.crop((left, top, right, bottom))

                img.save(fname)
        except:
            continue

output = '\n'.join(breeds)
with open('../Classes/breeds.txt', 'w') as f:
    f.write(output)

