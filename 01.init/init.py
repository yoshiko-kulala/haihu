import sys
from PIL import Image
import tensorflow as tf
import chainer

print("python\n"+sys.version+"\n")
print("PIL\n"+Image.__version__+"\n")
print("TF\n"+tf.__version__+"\n")
print("chainer\n"+chainer.__version__+"\n")