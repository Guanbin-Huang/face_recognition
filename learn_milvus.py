"""
Milvus的操作手册，请参考(以官网为主)：
    官网：https://www.milvus.io/cn/docs/v0.10.2/example_code.md
    API参考Python：https://github.com/milvus-io/pymilvus
        pip install pymilvus
    Milvus用户手册：bookstack.cn/read/milvus-0.6-zh/guides-milvus_operation.md
1.连接Milvus服务端(创建客户端)、关闭客户端
    虚拟机中用Docker容器启动了Milvus服务器，对外暴露端口：tcp协议19530，http协议19121。
    sudo docker run -d --name milvus_cpu_0.10.2 -p 19530:19530 -p 19121:19121 ....
    因此这里连接Milvus服务的HOST是虚拟机的IP地址，PORT是19530。
    在Ubuntu中使用ifconfig可以看到虚拟机的IP。

    连接服务端代码示例，localhost改为虚拟机的IP：
    from milvus import Milvus, IndexType, MetricType, Status

    # 连接Milvus服务端，即创建客户端
    milvus = Milvus(host='localhost', port='19530')
    或
    milvus = Milvus(uri='tcp://localhost:19530')

    # 关闭客户端
    milvus.close()
2.创建、删除集合
    1）创建集合名为 test01，维度为 256，自动创建索引的数据文件大小为 1024 MB，距离度量方式为欧氏距离（L2）的集合：
    metric_type是计算向量距离的方式，MetricType.L2求欧式距离，MetricType.IP求內积。
    可见这里存储的是归一化后的向量，如此欧式距离和內积才可以作为向量相似度的度量。

    # 创建集合所需参数
    param = {
        'collection_name':'test01',     # 集合名
        'dimension':256,                # 维度
        'index_file_size':1024,         # 自动创建索引的数据文件大小
        'metric_type':MetricType.L2     # 距离度量方式
    }
    milvus.create_collection(param)

    2）删除集合: 根据集合名删除集合
    milvus.drop_collection(collection_name='test01')
2.创建、删除分区
    1）创建分区：你可以通过标签将集合分割为若干个分区，从而提高搜索效率。每个分区实际上也是一个集合。
        milvus.create_partition(collection_name='test01', partition_tag='tag01')
    2）删除分区
        milvus.drop_partition(collection_name='test01', partition_tag='tag01')
3.插入、删除向量
    1)在集合中插入向量
        import random

        # 随机生成20个256维的向量
        vectors = [[random.random() for _ in range(256)] for _ in range(20)]

        # 插入向量列表，返回插入状态和向量的id列表。如果你不指定向量 ID，Milvus 自动为向量分配 ID。
        status, ids = milvus.insert(collection_name='test01', records=vectors)

        # 也可以自定义向量ID:
        vector_ids = [id for id in range(20)]
    2)在分区中插入向量
        milvus.insert('test01', vectors, partition_tag="tag01")
    3)通过ID删除向量
        ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        milvus.delete_entity_by_id(collection_name='test01', id_array=ids)
4.创建、删除索引
    目前，一个集合只支持一种索引类型，切换索引类型会自动删除旧的索引文件。
    create_index() 会指定该集合的索引类型，并同步为之前插入的数据建立索引，
    后续插入的数据在大小达到 index_file_size 时，索引会在后台自动建立。
    在实际生产环境中，如果是流式数据，建议在插入向量之前先创建索引，以便后续系统自动建立；
    如果是静态数据，建议导入所有数据后再一次性创建索引。
    1）创建索引: 在创建其它索引前，FLAT 作为集合的默认索引类型。
        # 准备创建索引所需参数。对于不同的索引类型，创建索引所需参数也有区别。
        ivf_param = {'nlist': 16384}
        # 为指定集合创建索引
        status = milvus.create_index('test01', IndexType.IVF_FLAT, ivf_param)
    2）删除索引: 删除索引后，集合再次使用默认索引类型 FLAT。
        milvus.drop_index('test01')
5.查询向量: Milvus支持在集合或分区中查询向量
    1）在集合中查询向量
        # 创建搜索参数。搜索参数是一个 JSON 字符串，在 Python SDK 中以字典来表示。
        # 对于不同的索引类型，搜索所需参数也有区别。所有的搜索参数都必须赋值。
        search_param = {'nprobe': 16}

        # 创建随机向量作为 query_records 进行搜索：
        # top_k 是与目标向量最相似的 k 条向量，在搜索时定义。top_k 的取值范围是 (0, 2048]。
        q_records = [[random.random() for _ in range(256)] for _ in range(5)]
        milvus.search(collection_name='test01', query_records=q_records, top_k=2, params=search_param)
    2）在分区中查询向量
        q_records = [[random.random() for _ in range(256)] for _ in range(5)]
        milvus.search(collection_name='test01', query_records=q_records, top_k=1, partition_tags=['tag01'], params=search_param)
6.数据落盘
    对有关数据进行更改操作时，通过flush操作将集合中的数据从内存落盘。。
    Milvus也会执行自动落盘。自动落盘会在固定时间周期(1秒)将现存集合的数据进行落盘操作。
    在调用 delete 接口后，用户可以选择再调用 flush，保证新增的数据可见，被删除的数据不会再被搜到。

    milvus.flush(collection_name_array=['test01'])
7.数据段整理
    数据段是Milvus自动将插入的向量数据合并所获得的数据文件。一个集合可包含多个数据段。
    如果一个数据段中的向量数据被删除，被删除的向量数据占据的空间并不会被自动释放。
    因此，我们可以对集合中的数据段进行compact操作以释放多余空间。

    # compact操作可释放多余空间(即释放被删除数据所占据的数据段空间)。
    milvus.compact(collection_name='test01', timeout='1')
"""
# This program demos how to connect to Milvus vector database,
# create a vector collection,
# insert 10 vectors,
# and execute a vector similarity search.

import random
import numpy as np

from milvus import Milvus, IndexType, MetricType, Status

# Milvus服务的IP地址和端口
_HOST = '192.168.25.29'
_TCP_PORT = '19530'  # 默认(TCP)端口
_HTTP_PORT = '19121'  # 默认的HTTP端口

# 向量参数
_DIM = 8  # 向量的维度

_INDEX_FILE_SIZE = 32  # max file size of stored index


def main():
    # 1.连接Milvus服务端，即创建客户端(连接池)。
    milvus = Milvus(_HOST, _TCP_PORT)  # TCP
    # milvus = Milvus(_HOST, _HTTP_PORT, handler="HTTP")  # HTTP

    # 2.创建集合
    # 集合名称
    collection_name = 'example_collection_vector'
    status = milvus.drop_collection(collection_name)
    print(status)
    # 判断该集合是否存在，如果不存在就创建该集合
    status, ok = milvus.has_collection(collection_name)
    print(status, ok)  # Status(code=0, message='Success') False
    if not ok:
        param = {
            'collection_name': collection_name,
            'dimension': _DIM,
            'index_file_size': _INDEX_FILE_SIZE,  # optional
            'metric_type': MetricType.L2  # optional
            # 'metric_type': MetricType.IP  # optional
        }
        # 创建集合
        milvus.create_collection(param)

    # 获取Milvus数据库中的所有集合名称列表
    _, collections = milvus.list_collections()
    print(len(collections), type(collections))  # 0 <class 'list'>

    # 获取某个指定集合(根据集合名称)的信息
    _, collection = milvus.get_collection_info(collection_name)
    print(collection)

    # 3.创建向量vectors，并将vectors插入到集合中。
    # 1）创建vectors，vectors形如(N,dim_num)。N是向量的个数，dim_num是向量的维度。
    # 注意：每个维度的值必须是float32类型。
    vectors = [[random.random() for _ in range(_DIM)] for _ in range(20)]
    print(len(vectors), type(vectors), type(vectors[0][0]))  # 10 <class 'list'> <class 'float'>
    print(vectors)
    # 也可以使用numpy来创建vectors:
    # vectors_numpy = np.random.rand(20, _DIM).astype(np.float32)
    # # 10000 <class 'numpy.ndarray'> (10000, 8) float32
    # print(len(vectors_numpy), type(vectors_numpy), vectors_numpy.shape, vectors_numpy.dtype)
    # vectors = vectors_numpy

    # 2）将vectors插入到集合中，返回插入状态status和vectors的id列表
    status, ids = milvus.insert(collection_name=collection_name, records=vectors)
    print(ids)
    # 也可以自定义向量ID:
    # vector_ids = [id for id in range(20)]
    # status, ids = milvus.insert(collection_name=collection_name, records=vectors, ids=vector_ids)
    if not status.OK():
        print("Insert failed: {}".format(status))

    # 4.其他操作
    # 1）数据落盘: 通过flush操作将更改的集合数据从内存中落盘。
    # Milvus会自动落盘，自动落盘周期是1秒。
    # 根据集合名称查看存储的向量个数。
    status, result = milvus.count_entities(collection_name)
    print(result)  # 如果不设置断点，这里向量数量为0。
    # 手动落盘：将集合中的数据从内存落盘。
    milvus.flush([collection_name])
    # 根据集合名称查看存储的向量个数
    status, result = milvus.count_entities(collection_name)
    print(result)  # 40

    # 2）获取目前某个集合的统计信息
    _, info = milvus.get_collection_stats(collection_name)
    print(info)

    # 3）根据向量ids获取原始向量
    status, result_vectors = milvus.get_entity_by_id(collection_name, ids[:10])
    print(result_vectors)

    # 4）创建向量的索引，可以让检索向量更快速。
    # 准备创建索引所需参数
    index_param = {
        'nlist': 2048
    }
    print("Creating index: {}".format(index_param))
    status = milvus.create_index(collection_name, IndexType.IVF_FLAT, index_param)
    print(status)

    # 5）获取指定集合的索引描述信息: 包括集合名、索引类型、索引参数
    status, index = milvus.get_index_info(collection_name)
    print(index)  # Status(code=0, message='Build index successfully!')

    # 5.向量相似度查询
    # 使用前10个向量进行相似度查询
    query_vectors = vectors[0:10]
    # 搜索参数(不同的索引类型，搜索所需参数也有区别)
    search_param = {
        "nprobe": 16
    }
    print("Searching ... ")
    # search()函数的参数：如果你不指定 partition_tags， Milvus 会在整个集合中搜索。
    param = {
        'collection_name': collection_name,
        'query_records': query_vectors,  # 搜索向量/目标向量
        'top_k': 1,  # top_k 是与目标向量最相似的 k 条向量
        'params': search_param,
    }
    # 在集合中搜索与目标向量query_vectors最相似的top_k条向量。
    status, results = milvus.search(**param)
    print(len(results), type(results[0]), type(results[0][0]))
    if status.OK():
        # 显示搜索结果
        if results[0][0].distance == 0.0 or results[0][0].id == ids[0]:
            print('Query result is correct')
        else:
            print('Query result isn\'t correct')

        # 打印搜索结果
        print(results)
    else:
        print("Search failed. ", status)

    # 6.删除集合
    status = milvus.drop_collection(collection_name)
    print(status)  # Status(code=0, message='Delete collection successfully!')

    # 7.关闭客户端(连接池置空): 这个可以不写，因为Milvus的析构函数__del__中已经调用self.close()
    milvus.close()


if __name__ == '__main__':
    main()
