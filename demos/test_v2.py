import tensorflow as tf


def string_input(filenames_file):
    input_queue = tf.train.string_input_producer([filenames_file], shuffle=False)
    line_reader = tf.TextLineReader()
    # 获取文件中的每行的内容
    _, line = line_reader.read(input_queue)
    split_line = tf.string_split([line]).values
    # record_defaults：指定每一个样本的每一列的类型，指定默认值
    records = [["None"], ["Node"]]
    # 有几列就用几个参数接收
    # example = tf.decode_csv(line, records)
    # example1 = tf.decode_csv(split_line[0], records)
    # example2 = tf.decode_csv(split_line[1], records)
    example1, example2 = tf.decode_csv(line, records, " ")
    # 取出图片名


    print("========================================================")
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        # 启动计算图中所有的队列线程 调用tf.train.start_queue_runners来将文件名填充到队列，否则read操作会被阻塞到文件名队列中有值为止。
        threads = tf.train.start_queue_runners(coord=coord)
        # 主线程，消费50个数据
        for _ in range(2):
            # print(sess.run(example))
            example1_t, example2_t = sess.run([example1, example2])
            print(example1_t)
            print(example2_t)
        # 主线程计算完成，停止所有采集数据的进程
        coord.request_stop()
        # 指定等待某个线程结束
        coord.join(threads)


def build_dataset(filenames_file):
    # Start with a dataset of filenames.
    # Use Dataset.flat_map() and tf.data.TextLineDataset to convert the
    # filenames into a dataset of lines.
    dataset = tf.data.TextLineDataset([filenames_file])
    dataset = dataset.map(deal_image)
    # iterator = dataset.make_one_shot_iterator()
    # x = iterator.get_next()
    # split_line = tf.string_split([x]).values

    iterator = dataset.make_one_shot_iterator()
    left, right = iterator.get_next()
    with tf.Session() as sess:
        for i in range(2):
            print(sess.run([left, right]))

def deal_image(value):
    data_path = '/Users/icodeboy/PycharmProjects'
    records = [["None"], ["Node"]]
    pic_left, pic_right = tf.decode_csv(value, records, " ")



    return pic_left, pic_right

def main():
    print("start replace")
    filenames_file = '/Users/icodeboy/PycharmProjects/monodepth/demos/filenames/kitti_train_files.txt'
    print(filenames_file)
    build_dataset(filenames_file)
    # string_input(filenames_file)


if __name__ == '__main__':
    main()
