# 相似搜索引擎Milvus
"""
1.不同的人脸特征，存储在不同的集合中。
"""
from milvus import Milvus, IndexType, MetricType

# 全局常量
# _HOST = '192.168.25.29'  # Milvus服务的IP
_HOST = '192.168.203.203'  # Milvus服务的IP
_PORT = '19530'  # Milvus服务的(TCP)端口
_DIM = 1024  # 向量的维度
# _DIM = 1024  # 向量的维度
_INDEX_FILE_SIZE = 32  # 数据段的大小


class MilvusEngine:
    # 初始化
    def __init__(self):
        super(MilvusEngine, self).__init__()
        # 创建客户端(连接Milvus服务端)
        self.client = Milvus(_HOST, _PORT)
        print("Mlivus open success")

    # 保存向量
    def save(self, collection_name, vectors):
        # 1.创建集合
        _, ok = self.client.has_collection(collection_name)
        if not ok:
            # 1）创建集合
            param = {
                'collection_name': collection_name,
                'dimension': _DIM,
                'index_file_size': _INDEX_FILE_SIZE,  # optional
                'metric_type': MetricType.IP  # 向量归一化后的內积与余弦相似度等价
            }
            self.client.create_collection(param)

            # 创建集合向量的索引。为向量添加索引可以让索引速度更快。
            # 因为是流式数据，因此在插入向量之前先指定索引类型，以便后续系统自动建立。
            # 准备创建索引所需参数
            index_param = {
                'nlist': 2048  # nlist是聚类单元数，范围[1, 65536]
            }
            # print("Creating index: {}".format(index_param))
            # 指定该集合的索引类型，并同步为之前插入的数据建立索引。
            self.client.create_index(collection_name, IndexType.IVF_FLAT, index_param)

        # 2.向集合中插入向量
        self.client.insert(collection_name, vectors)

        # 3.手动落盘：将集合中的数据从内存落盘。
        self.client.flush([collection_name])

    # 向量相似度检索
    def search(self, query_vectors, collection_name):
        # 检索参数
        search_param = {
            "nprobe": 16
        }
        # print("Searching ... ")
        # 不指定partition_tags表示在集合的所有分区中检索
        param = {
            'collection_name': collection_name,
            'query_records': query_vectors,  # 搜索向量/目标向量
            'top_k': 1,  # top_k 是与目标向量最相似的 k 条向量
            'params': search_param,
        }
        # 在集合中搜索与目标向量query_vectors最相似的top_k条向量。
        status, results = self.client.search(**param)

        return results[0][0].distance

    def drop(self, collection_name):
        status = self.client.drop_collection(collection_name=collection_name)
        return status

    # 获取数据库中的所有集合名称列表
    def list(self):
        _, collections = self.client.list_collections()
        return collections

    # 清空数据库中的所有集合
    def clear(self):
        # 1.获取数据库中的所有集合名称列表
        collections = self.list()

        # 2.遍历集合名称列表，执行删除集合操作
        for collection_name in collections:
            self.client.drop_collection(collection_name)

    def get(self, id):
        status, vector = self.client.get_entity_by_id(collection_name='shituo', ids=[id])
        return vector


if __name__ == '__main__':
    e = MilvusEngine()
    print(e.list())
    # e.clear()
    # print(e.get(25))
    print(e.drop('SLS'))