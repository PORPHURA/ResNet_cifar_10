'''
    ResNet-101 v2 validate on CIFAR-10
'''


import numpy as np
import os
import tensorflow as tf
import cifar_input

from tensorflow.contrib.slim import nets

slim = tf.contrib.slim

# FLAGS参数设置
FLAGS = tf.app.flags.FLAGS
# 数据集名称
tf.app.flags.DEFINE_string('dataset',
                           'cifar10',
                           'Name of dataset')
# # 测试数据路径
tf.app.flags.DEFINE_string('eval_data_path',
                           'data/cifar-10-batches-bin/test_batch.bin',
                           'Filepattern for eval data')
# 图片尺寸
tf.app.flags.DEFINE_integer('image_size',
                            32,
                            'Image side length.')
# 分类数目
tf.app.flags.DEFINE_integer('num_classes', 10, 'Number of classes')
# 测试数据的Batch数量
tf.app.flags.DEFINE_integer('batch_size',
                            100,
                            'Size of batches.')
# 训练模型路径
tf.app.flags.DEFINE_string('checkpoint_model_path',
                           'model/',
                           'Path of ResNet-101')
# 测试轮数
tf.app.flags.DEFINE_integer('num_steps', 20, 'Number of training steps')


def main(_):
    inputs, labels = cifar_input.build_input(FLAGS.dataset, FLAGS.eval_data_path, FLAGS.batch_size, 'eval')
    is_training = True

    with slim.arg_scope(nets.resnet_v2.resnet_arg_scope()):
        net, endpoints = nets.resnet_v2.resnet_v2_101(inputs, num_classes=None,
                                                      is_training=is_training)
    with tf.variable_scope('Logits'):
        net = tf.squeeze(net, axis=[1, 2])
        net = slim.dropout(net, keep_prob=0.5, scope='scope')
        logits = slim.fully_connected(net, num_outputs=FLAGS.num_classes, activation_fn=None,
                                      scope='fc')

    # 有选择地恢复变量
    checkpoint_exclude_scopes = 'Logits'
    exclusions = None
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
            if not excluded:
                variables_to_restore.append(var)

    logits = tf.nn.softmax(logits)
    classes = tf.argmax(logits, axis=1, name='classes')
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(classes, dtype=tf.int32), labels), dtype=tf.float32))

    # 获取最新的模型
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_model_path)
    saver_restore = tf.train.Saver()

    with tf.Session() as sess:
        # 载入训练模型
        saver_restore.restore(sess, ckpt.model_checkpoint_path)

        # 开启队列
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        correct_prediction = 0

        for i in range(FLAGS.num_steps):
            correct_prediction += sess.run(accuracy)

        # 输出测试情况
        precision = correct_prediction/FLAGS.num_steps
        validate_log = 'Validation precision: {:.4f}'.format(precision)
        print(validate_log)
        # 关闭队列
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
