"""
    CIFAR dataset input module.
"""

import tensorflow as tf


def build_input(dataset, data_path, batch_size, mode):
    """Build CIFAR image and labels.

    Args:
    dataset(数据集): Either 'cifar10' or 'cifar100'.
    data_path(数据集路径): Filename for data.
    batch_size: Input batch size.
    mode(模式）: Either 'train' or 'eval'.
    Returns:
    images(图片): Batches of images. [batch_size, image_size, image_size, 3]
    labels(类别标签): Batches of labels. [batch_size]

    """
  
    # 数据集参数
    image_size = 32
    if dataset == 'cifar10':
        label_bytes = 1
        label_offset = 0
    else:
        raise ValueError('Not supported dataset %s', dataset)

    # 数据读取参数
    depth = 3
    image_bytes = image_size * image_size * depth
    record_bytes = label_bytes + label_offset + image_bytes

    # 获取文件名列表
    data_files = tf.gfile.Glob(data_path)
    # 文件名列表生成器
    file_queue = tf.train.string_input_producer(data_files, shuffle=True)
    # 文件名列表里读取原始二进制数据
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, value = reader.read(file_queue)

    # 将原始二进制数据转换成图片数据及类别标签
    record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
    label = tf.cast(tf.slice(record, [label_offset], [label_bytes]), tf.int32)
    # 将数据串 [depth * height * width] 转换成矩阵 [depth, height, width].
    depth_major = tf.reshape(tf.slice(record, [label_bytes], [image_bytes]),
                             [depth, image_size, image_size])
    # 转换维数：[depth, height, width]转成[height, width, depth].
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    if mode == 'train':
        # 大小调整
        image = tf.image.resize_image_with_crop_or_pad(
            image, image_size + 4, image_size + 4)
        image = tf.random_crop(image, [image_size, image_size, 3])
        # 随机翻转
        image = tf.image.random_flip_left_right(image)
        # 亮度标准化
        image = tf.image.per_image_standardization(image)

        # 创建随机队列
        example_queue = tf.RandomShuffleQueue(
            capacity=16 * batch_size,
            min_after_dequeue=8 * batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[image_size, image_size, depth], [1]])
        num_threads = 16
    else:
        # 大小调整
        image = tf.image.resize_image_with_crop_or_pad(
            image, image_size, image_size)
        # 亮度标准化
        image = tf.image.per_image_standardization(image)

        example_queue = tf.FIFOQueue(
            3 * batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[image_size, image_size, depth], [1]])
        num_threads = 1

    # 数据入队操作
    example_enqueue_op = example_queue.enqueue([image, label])
    # 队列执行器
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(example_queue, [example_enqueue_op] * num_threads))

    # 数据出队操作，从队列读取Batch数据
    images, labels = example_queue.dequeue_many(batch_size)
    labels = tf.squeeze(labels, axis=1)

    # init = tf.global_variables_initializer()

    # with tf.Session() as sess:
    #     sess.run(init)
    #     tf.train.start_queue_runners(sess)
    #     images_, labels_ = sess.run([images, labels])
    #     print('Batch extracted')

    return images, labels

#
# def main(_):
#     images, labels = build_input('cifar10', 'data/cifar-10-batches-bin/data_batch*', 128, 'train')
#     print('Success')
#
#
# if __name__ == '__main__':
#     tf.app.run()

