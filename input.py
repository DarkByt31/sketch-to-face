import imghdr
import os
import tensorflow as tf

# Read tfrecords and convert it into image files
class TFRecordExtractor:
	def __init__(self, tfrecord_dir, batch_size, epochs, image_height, image_width, shuffle_size):
		(_, _, filenames) = next(os.walk(tfrecord_dir))
		filenames = [tfrecord_dir + name for name in filenames]
		self.record_names = filenames
		self.batch_size = batch_size
		self.epochs = epochs
		self.image_height = image_height
		self.image_width = image_width
		self.shuffle_size = shuffle_size

	def _extract_fn(self, tfrecord):
		features = {
			# 'filename': tf.FixedLenFeature([], tf.string),
			'image': tf.FixedLenFeature([], tf.string),
		}

		# Extract the data record
		sample = tf.parse_single_example(tfrecord, features)
		image = tf.image.decode_jpeg(sample['image'], channels=3)
		image = tf.image.convert_image_dtype(image, tf.float32)
		image = tf.image.resize_images(image, [self.image_height, self.image_width])

		return image

	def extract_image(self):
		# Create folder to store extracted images
		# os.makedirs(img_out_dir, exist_ok=True)

		# Pipeline of dataset and iterator
		dataset = tf.data.TFRecordDataset(self.record_names)
		dataset = dataset.shuffle(self.shuffle_size, reshuffle_each_iteration=True)
		dataset = dataset.map(self._extract_fn)
		dataset = dataset.batch(self.batch_size)
		dataset = dataset.repeat(self.epochs)

		# create one shot iterator
		iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
		next_image_data = iterator.get_next()

		return next_image_data, ""

class old_input:
	def is_image_valid(filepath):
		return imghdr.what(filepath) is not None

	def get_image_paths(image_dir):
		image_paths = []
		for root, directories, filenames in os.walk(image_dir):
			print("filenames: " + str(filenames[:100]))
			image_paths += [os.path.join(root, filename) for filename in filenames]
		image_paths = [filepath for filepath in image_paths if self.is_image_valid(filepath)]

		return image_paths


	def inputs(image_dir, batch_size, min_queue_examples, input_height, input_width):
		def read_images(image_paths):
			filename_queue = tf.train.string_input_producer(image_paths)
			reader = tf.WholeFileReader()
			key, value = reader.read(filename_queue)
			image = tf.image.decode_image(value)
			image = tf.image.convert_image_dtype(image, dtype=tf.float32)
			# understand this line
			image.set_shape([None, None, 3])

			return image

		image_paths = self.get_image_paths(image_dir)
		images = read_images(image_paths)
		images = tf.image.crop_to_bounding_box(images, 30, 0, 178, 178)
		# images = tf.image.random_flip_left_right(images)
		images = tf.image.resize_images(images, [input_height, input_width])

		total_image_count = len(image_paths)
		input_batch = tf.train.shuffle_batch([images],
											 batch_size=batch_size,
											 num_threads=16,
											 capacity=min_queue_examples + 3 * batch_size,
											 min_after_dequeue=min_queue_examples)

		return input_batch, total_image_count


def inputs(file_dir, batch_size, min_queue_examples, input_height, input_width, tfrecord=False, epochs=10, shuffle_size=1000):
	if(tfrecord):
		t = TFRecordExtractor(file_dir, batch_size, epochs, input_height, input_width, shuffle_size)
		return t.extract_image()
	else:
		return old_input.inputs(file_dir, batch_size, min_queue_examples, input_height, input_width)

if __name__ == '__main__':
	pass
