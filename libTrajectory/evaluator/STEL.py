import logging

import faiss


class Evaluator(object):

    def cosine_top(self, vector1, vector2, topn=5):
        logging.info("evaluate")
        admin = vector1.shape[1]
        indexIP = faiss.IndexFlatIP(admin)
        faiss.Index
        indexIP.add(vector1)
        distances, indices = indexIP.search(vector2, topn)
        # 索引相同者为正样本
        print(distances)
        print(indices)
