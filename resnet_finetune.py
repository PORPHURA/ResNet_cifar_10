'''
    ResNet-101 v2 finetune on CIFAR-10
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
# 训练数据路径
tf.app.flags.DEFINE_string('train_data_path',
                           'data/cifar-10-batches-bin/data_batch*',
                           'Filepattern for training data.')
# 测试数据路径
tf.app.flags.DEFINE_string('eval_data_path',
                           'data/cifar-10-batches-bin/test_batch.bin',
                           'Filepattern for eval data')
# 图片尺寸
tf.app.flags.DEFINE_integer('image_size',
                            32,
                            'Image side length.')
# 训练数据的Batch数量
tf.app.flags.DEFINE_integer('batch_size',
                            128,
                            'Size of batches.')
# 预训练模型路径
tf.app.flags.DEFINE_string('resnet_model_path',
                           'resnet_v2_101/resnet_v2_101.ckpt',
                           'Path of ResNet-101')
# 模型存储路径
tf.app.flags.DEFINE_string('model_save_path',
                           'model/',
                           'Directory to keep the checkpoints.')
# 分类数目
tf.app.flags.DEFINE_integer('num_classes', 10, 'Number of classes')
# 训练步数
tf.app.flags.DEFINE_integer('num_steps', 5000, 'Number of training steps')


def main(_):
    inputs, labels = cifar_input.build_input(FLAGS.dataset, FLAGS.train_data_path, FLAGS.batch_size, 'train')
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

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(losses)

    logits = tf.nn.softmax(logits)
    classes = tf.argmax(logits, axis=1, name='classes')
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(tf.cast(classes, dtype=tf.int32), labels), dtype=tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train_step = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    # 设置恢复、保存
    saver_restore = tf.train.Saver(var_list=variables_to_restore)
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        sess.run(init)

        # 载入预训练模型
        saver_restore.restore(sess, FLAGS.resnet_model_path)
        # 开启队列
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(FLAGS.num_steps):
            sess.run(train_step)
            loss_, acc_ = sess.run([loss, accuracy])

            # 输出训练情况
            train_log = 'Step: {}, Loss: {:.4f}, Accuracy: {:.4f}'.format(i+1, loss_, acc_)
            print(train_log)

            # 每50次循环保存一次模型
            if (i+1) % 50 == 0:
                saver.save(sess, FLAGS.model_save_path, global_step=i+1)
                print('Model saved.')
        # 关闭队列
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
