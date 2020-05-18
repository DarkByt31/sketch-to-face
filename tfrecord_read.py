import tensorflow as tf
import os
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.image as mpimg

epochs = 3
batch_size = 4
shuffle_size = 50
img_out_dir = "images/record_out/"
record_dir = "images/records/"
files_per_record = 110
num_records = 2
input_height = 256
input_width = 256

class TFRecordExtractor:
	def __init__(self, tfrecord_dir):
		(_, _, filenames) = next(os.walk(tfrecord_dir))
		filenames = [tfrecord_dir + name for name in filenames]
		self.record_names = filenames

	def _extract_fn(self, tfrecord):
		# Extract features using the keys set during creation
		features = {
			'filename': tf.FixedLenFeature([], tf.string),
			'image': tf.FixedLenFeature([], tf.string),
		}

		# Extract the data record
		samples = tf.parse_single_example(tfrecord, features)

		image = samples['image']
		image = tf.image.decode_jpeg(image, channels=3)
		image = tf.image.convert_image_dtype(image, tf.float32)
		image = tf.image.resize_images(image, [256, 256])

		filename = samples['filename']
		return [image, filename]

	def extract_image(self):
		# Create folder to store extracted images
		os.makedirs(img_out_dir, exist_ok=True)

		# Pipeline of dataset and iterator
		dataset = tf.data.TFRecordDataset(self.record_names)
		dataset = dataset.shuffle(shuffle_size, reshuffle_each_iteration=True)
		dataset = dataset.map(self._extract_fn)
		dataset = dataset.batch(batch_size)
		dataset = dataset.repeat(epochs)
		iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
		next_image_data = iterator.get_next()

		iterations = num_records * files_per_record * epochs / batch_size
		with tf.compat.v1.Session() as sess, tqdm(total = iterations) as pbar:
			sess.run(tf.global_variables_initializer())

			for e in range(epochs):
				idir = img_out_dir + str(e) + '/'
				os.makedirs(idir, exist_ok=True)
				print("epoch: ", e)
				for i in range(0, num_records * files_per_record, batch_size):
					# images is a list of size batch_size
					images, filenames = sess.run(next_image_data)
					for j in range(batch_size):	mpimg.imsave(idir + str(i+j) + '.jpg', images[j])
					pbar.update(batch_size)

if __name__ == '__main__':
	t = TFRecordExtractor(record_dir)
	t.extract_image()
