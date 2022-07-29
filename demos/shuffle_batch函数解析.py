def shuffle_batch(tensors, batch_size, capacity, min_after_dequeue,
                  num_threads=1, seed=None, enqueue_many=False, shapes=None,
                  allow_smaller_final_batch=False, shared_name=None, name=None):


# Creates batches by randomly shuffling tensors. 通过随机乱序张量来创建batch
这个函数将以下内容添加到当前的‘Graph’中
