# MIT License
# 
# Copyright (c) 2018 H. Watanabe 
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from PIL import Image
import chainer

def save_test(data, index, num):
    img = Image.new("L", (28, 28))
    pix = img.load()
    for i in range(28):
        for j in range(28):
            pix[i, j] = int(data[i+j*28]*256)
    img2 = img.resize((280, 280))
    filename = str(num) + "/test" + "{0:06d}".format(index) + ".png"
    img2.save(filename)
    print(filename)
    
def save_train(data, index, num):
    img = Image.new("L", (28, 28))
    pix = img.load()
    for i in range(28):
        for j in range(28):
            pix[i, j] = int(data[i+j*28]*256)
    img2 = img.resize((280, 280))
    filename = str(num) + "/train" + "{0:06d}".format(index) + ".png"
    img2.save(filename)
    print(filename)

def main():
    train, test = chainer.datasets.get_mnist()
    for i in range(10):
        dirname = str(i)
        if os.path.isdir(dirname) is False:
            os.mkdir(dirname)
    for i in range(len(test)):
        save_test(test[i][0], i, test[i][1])
#     for i in range(len(train)):
#         save_train(train[i][0], i, train[i][1])

if __name__ == '__main__':
    main()