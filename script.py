'''
cedrick argueta

run this file inside the unzipped directory
'''

import os

folders = os.listdir('.')
categories = os.listdir('./photo/tx_000000000000/')
print(categories)
for folder in folders:
    if not os.path.isdir(folder): continue
    for cat in categories:
        try:
            os.mkdir(folder + '/' + cat)
        except OSError:
            pass
    for subdir in os.listdir(folder):
        if subdir in categories: continue
        for cat in os.listdir(folder + '/' + subdir + '/'):
            for f in os.listdir(folder + '/' + subdir + '/' + cat + '/'):
                os.rename(folder + '/' + subdir + '/' + cat + '/' + f, folder + '/' + cat + '/' + f[:-4] + '_' + subdir + '.png') 
