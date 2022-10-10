from i2v.base import Illustration2VecBase
from i2v.chainer_i2v import ChainerI2V, make_i2v_with_chainer

from PIL import Image

illust2vec = make_i2v_with_chainer(
    "i2v/illust2vec_tag_ver200.caffemodel", "i2v/tag_list.json")

img = Image.open("dataset/exp-data/100-0.jpg")
print(illust2vec.estimate_plausible_tags([img], threshold=0.5))
