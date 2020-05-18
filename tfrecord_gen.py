import tensorflow as tf
import os
import io
from PIL import Image
from tqdm import tqdm
import argparse
#import matplotlib.image as mpimg

img_input_dir = "images/train/img_align_celeba/"
record_dir = "images/records/"
train_size = 10
train_records = 2
image_height = 256
image_width = 256
create_test = False
test_size = 100

# record output name: img1.tfrecord
class GenerateTFRecord:
	def convert_image_folder(self, img_folder, tfrecord_dir_name):
		# Get all file names of images present in folder
		img_paths = os.listdir(img_folder)
		img_paths = [os.path.abspath(os.path.join(img_folder, img)) for img in img_paths]
		os.makedirs(tfrecord_dir_name, exist_ok=True)
		dir_name = tfrecord_dir_name
		if(create_test):
			os.makedirs(tfrecord_dir_name + 'train/', exist_ok=True)
			os.makedirs(tfrecord_dir_name + 'test/', exist_ok=True)
			dir_name = tfrecord_dir_name + 'train/'

		k = 0
		iterations = train_size
		if(create_test):	iterations = iterations + test_size
		print("len(img_paths): ", len(img_paths))
		print("converting images: ")
		pbar = tqdm(total=iterations)
		for i in range(train_records):
			tqdm.write("writing in record: " str(i+1))
			with tf.io.TFRecordWriter(dir_name + 'img' + str(i+1) + '.tfrecord') as writer:
				while( k < min(int(train_size * (1.0+i)/train_records), len(img_paths)) ):
					# tqdm.write(k, ": " + img_paths[k])
					example = self._convert_image(img_paths[k])
					writer.write(example.SerializeToString())
					k += 1
					pbar.update()

		if(create_test):
			tqdm.write("creating test record. k = " + str(k))
			with tf.io.TFRecordWriter(tfrecord_dir_name + 'test/' + 'img1' + '.tfrecord') as writer:
				while( k < iterations ):
					# tqdm.write(k, ": " + img_paths[k])
					example = self._convert_image(img_paths[k])
					writer.write(example.SerializeToString())
					k += 1
					pbar.update()

		pbar.close()

	def _convert_image(self, img_path):
		filename = os.path.basename(img_path)

		# Read image data in terms of bytes
		# with open(img_path, 'rb') as fid:
		# 	image_data = fid.read()
		image_data = self.preprocess(img_path)

		example = tf.train.Example(features = tf.train.Features(feature = {
			'filename': tf.train.Feature(bytes_list = tf.train.BytesList(value = [filename.encode('utf-8')])),
			'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_data])),
		}))
		return example

	def preprocess(self, img_path):
		image = Image.open(img_path)
		image = image.crop((30, 0, 178, 178))
		image = image.resize((image_height, image_width))

		buf = io.BytesIO()
		image.save(buf, format='JPEG')
		byte_img = buf.getvalue()

		return byte_img

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-create_test", action="store_true")
	parser.add_argument("--img_input_dir", help="input directory of images", default="images/train/img_align_celeba/")
	parser.add_argument("--record_dir", help="output directory for tfrecords", default="images/records/")
	parser.add_argument("--image_height", type=int, help="height of image after preprocessing", default=256)
	parser.add_argument("--image_width", type=int, help="width of image after preprocessing", default=256)
	parser.add_argument("--train_size", type=int, help="size of training data", default=10)
	parser.add_argument("--train_records", type=int, help="total number of training records", default=2)
	parser.add_argument("--test_size", type=int, help="size of test data", default=100)

	args = parser.parse_args()
	create_test = args.create_test
	img_input_dir =args.img_input_dir
	record_dir = args.record_dir
	image_height = args.image_height
	image_width = args.image_width
	train_size = args.train_size
	train_records = args.train_records
	test_size = args.test_size


	t = GenerateTFRecord()
	t.convert_image_folder(img_input_dir, record_dir)
