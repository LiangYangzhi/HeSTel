import faiss
import networkx as nx


def evaluate(array, ncentroids, niter, user_id_list, G_real):
    print("clustering")
    d = array.shape[1]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=True)
    kmeans.train(array)

    print('getting clustering results...')
    D, I = kmeans.index.search(array, k=1)
    community = {}
    for i in range(len(I)):
        c = I[i][0]
        if c not in community:
            community[c] = []
        community[c].append(user_id_list[i])

    community = [community[i] for i in community.keys()]
    print('calculating modularity...')
    modularity = nx.community.modularity(G_real, community)
    return modularity
