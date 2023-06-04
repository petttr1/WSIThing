from multiprocessing import Pool

shared = {}


class WSIQueue:
    def __init__(self):
        self.queue = []

    def extract_nodes(self, roots):
        nodes = []
        for root in roots:
            self.queue = [root]
            while self.queue:
                node = self.queue.pop(0)
                node_children = node.get_children()
                self.queue.extend(node_children)
                nodes.append(node)
        return nodes

    # def init_func(self, args):
    #     shared['arr'] = args['arr']
    #     shared['size'] = args['size']

    def process_nodes(self, nodes, callback):  # , init_args=None):
        # if init_args is None:
        with Pool(None) as p:
            try:
                result = p.map(callback, nodes)
            except:
                raise
        # pool = Pool(None)
        # else:
        #     pool = Pool(None, self.init_func, init_args)
        return result
