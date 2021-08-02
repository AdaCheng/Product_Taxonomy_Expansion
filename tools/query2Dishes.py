import os,math,tqdm
from tools import gzh
class query2Dishes:
    def __init__(self,num_layer:int,data_path:str,hypo_json_path:str,poi_json_path:str,spu_json_path:str,dishes_txt_path:str,cate_txt_path:str,poi_spu_filter_path:str,need_init_poi=True,poi_stop_num=10):
        '''
        初始化
        :param num_layer: 需要构建的taxonomy层数
        :param data_path: 数据集保存位置
        :param hypo_json_path: 下位词的json文件地址
        :param poi_json_path: poi的json文件地址
        :param spu_json_path: spu的json文件地址
        :param dishes_txt_path: 推荐菜大全的txt文件地址
        :param cate_txt_path: poi种类的txt文件地址
        :param poi_spu_filter_path: query和poi关联数据的txt文件地址
        :param need_init_poi: 是否需要初始化poi
        :param poi_stop_num: 处理多少个poi
        '''
        self.data_path = data_path
        self.num_layer = num_layer
        self.spu_json_path = spu_json_path
        self.hypo_json_path = hypo_json_path
        self.poi_json_path = poi_json_path
        self.dishes_txt_path = dishes_txt_path
        self.cate_txt_path = cate_txt_path
        self.poi_spu_filter_path = poi_spu_filter_path
        self.layer_node_path = os.path.join(data_path,'layer_node.json')
        self.query2poi_fgc_path = os.path.join(data_path,'poi_fgc_path.json')

        # spu文件初始化与导入
        # 判断spu有没有初始化过
        if 'spu自底向上撞推荐菜_最长字符串匹配.json' not in os.listdir(self.data_path):
            self.getLayer()
            self.bottmUpCollide()
        else:
            self.query2spu_dishes_path = os.path.join(self.data_path, 'spu自底向上撞推荐菜_最长字符串匹配.json')
        self.spu_dishes = gzh.readJson(self.query2spu_dishes_path)
        print('spu初始化完成')

        # poi文件初始化与导入
        self.zz()
        if need_init_poi:
            self.init_poi_dishes(stop_num=poi_stop_num)
        # 选择前c/a个type的推荐菜
        self.query2fgc = gzh.readJson(self.query2poi_fgc_path)
        self.poi = gzh.readJson(self.poi_json_path)
        print('poi初始化完成')
    # 获得query的poi的推荐菜
    def query2poi_dishes(self,query,a,c):
        '''
        通过query获得poi的推荐菜，a和c只是一个比例，取前c/a个店即可
        :param query:
        :param a: 总量
        :param c: 选择量
        :return:
        '''
        dishes = {}
        for shop in self.query2poi[query]:
            if shop not in self.poi:
                continue
            for dish in self.poi[shop]:
                dishes[dish] = dishes.get(dish,0) + 1

        num = math.ceil(len(self.query2fgc[query]) * c / a)
        return_dishes = []
        fgc = self.query2fgc[query][:num]
        for shop in self.query2poi[query]:
            if self.poi2fgc[shop] in fgc:
                return_dishes.extend([i for i in self.poi[shop]])
        return_dishes = list(set(return_dishes))
        return_dishes = [[i,dishes[i]]for i in return_dishes]
        return sorted(return_dishes,key=lambda x:x[1],reverse=True)
    # 初始化两个全局字典
    def zz(self):
        '''
        初始化两个与poi有关的字典
        poi2fgc：通过poi获得这个poi的细粒度的种类
        query2poi：通过query获得poi list
        :return:
        '''
        f = open(self.cate_txt_path, encoding='utf-8')
        # 每个poi有哪些分类
        self.poi2fgc = {}
        for line in f:
            poi_id, name_id, name, fgc_id, fgc, cgc_id, cgc = line.strip().split('\t')
            self.poi2fgc[poi_id] = fgc_id
        f = open(self.poi_spu_filter_path, encoding='utf-8')
        # query到poi的索引
        self.query2poi = {}
        for line in f:
            try:
                query, idd, name, num = line.strip().split('\t')
                li = self.query2poi.get(query, set())
                li.add(idd)
                self.query2poi[query] = li
            except:
                pass
        f.close()
    # poi文件初始化与导入
    def init_poi_dishes(self,stop_num = 10):
        '''
        初始化poi文件
        :param stop_num: 需要初始化多少个poi文件（如果需要初始化全部poi文件，则需要将stop_num设置为负数）
        :return:
        '''
        poi = gzh.readJson(gzh.poi_path)
        hypo = gzh.readJson(gzh.hypo_path)
        query2fgc_count = {}
        query2fgc = {}

        for index, query in enumerate(tqdm.tqdm(self.query2poi)):
            if query not in ['葡萄', '芝士葡萄', '金香葡萄', '鲜葡萄', '葡萄干']:
                continue
            if index == stop_num:
                break
            if query not in hypo:
                continue
            if query not in query2fgc_count:
                query2fgc_count[query] = {}
            ac = gzh.Trie(hypo[query])
            for shop in self.query2poi[query]:
                if shop not in self.poi2fgc:
                    continue
                for dish in poi[shop]:
                    txt = ac.search(dish)
                    if txt:
                        query2fgc_count[query][self.poi2fgc[shop]] = query2fgc_count[query].get(self.poi2fgc[shop], 0) + 1
            query2fgc_count[query] = sorted(query2fgc_count[query].items(), key=lambda x: x[1], reverse=True)
            query2fgc[query] = [i[0] for i in query2fgc_count[query]]
        gzh.toJson(query2fgc, self.query2poi_fgc_path)
    # 获得query的spu撞出来的推荐菜
    def query2spu_dishes(self,query:str):
        '''
        通过query获得spu撞出来的推荐菜
        :param query:
        :return:
        '''
        if query in self.spu_dishes:
            return sorted(self.spu_dishes[query],key=lambda x:x[1],reverse=True)
    # 导出重构的json
    def getLayer(self):
        '''
        计算hypo中每个结点处于哪一层，方面从下往上遍历
        :return:
        '''
        hypo = gzh.readJson(self.hypo_json_path)
        # 入度
        nodes = set(hypo.keys())

        num_layer = self.num_layer
        index = 1
        layer = {}
        print('hypo共有%d个结点'%len(nodes))
        for _ in range(num_layer):
            in_node = {}

            for node in nodes:
                in_node[node] = 0
            for key in nodes:
                if key not in hypo:
                    continue
                for entity in hypo[key]:
                    if entity in in_node:
                        in_node[key] += 1
            layer[index] = [i[0] for i in in_node.items() if i[1] == 0]
            nodes = [i[0] for i in in_node.items() if i[1] != 0]
            print('%d层共有%d个结点'%(index,len(layer[index])))
            index += 1
        gzh.toJson(layer,self.layer_node_path)
    # 自底向上spu撞推荐菜
    def bottmUpCollide(self):
        '''
        哎……乱七八糟一堆东西，反正实现了需求hhh
        :return:
        '''
        # 准备数据
        layer = gzh.readJson(self.layer_node_path)
        f = open(self.dishes_txt_path, encoding='utf-8')
        new_dishes = set()
        for line in f:
            new_dishes.add(line.split('\t')[0])
        f.close()
        spu = gzh.readJson(self.spu_json_path)

        for key in spu:
            li = []
            for d in spu[key]:
                li.append(d[0])
            spu[key] = li

        # 遍历几层
        #num_layer = 2
        i = self.num_layer
        find = {}
        print('总共有%d个有spu的概念，总共有%d份推荐菜。' % (len(spu), len(new_dishes)))
        ac = {}
        while (i != 0):
            find[i] = {}
            # ac自动机
            ac[i] = gzh.Trie()
            dishes = new_dishes
            # 建立spu到concept的索引
            hyper = {}
            count = []
            for key in layer[str(i)]:
                if key not in spu:
                    count.append(key)
                    continue
                for entity in spu[key]:
                    li = hyper.get(entity, [])
                    li.append(key)
                    hyper[entity] = li
            print(
                '当前是第%d层，剩余%d个推荐菜，hypo文件中该层有%d个概念结点,其中有%d个概念结点没有spu' % (i, len(dishes), len(layer[str(i)]), len(count)))
            start = time.time()
            ac[i].add_words(dishes)
            print('第%d层自动机构建完成,耗时%.1f' % (i, time.time() - start))
            for key in layer[str(i)]:
                num = 0
                find_num = 0
                if key in count:
                    continue
                for s in spu[key]:
                    num += 1
                    txt = ac[i].search(s)
                    if txt:
                        # spu撞到的推荐菜
                        find_num += 1
                        match = [j[0] for j in txt.items()]
                        for dish in match:
                            try:
                                new_dishes.remove(dish)
                            except:
                                pass
                            # 遍历这个spu的concept
                            for concept in hyper[s]:
                                if concept not in find[i]:
                                    find[i][concept] = {}
                                find[i][concept][dish] = find[i][concept].get(dish,0) + 1
            ac.pop(i)
            i -= 1
        print('剩余推荐菜%d'%len(new_dishes))
        del new_dishes
        # 最长匹配
        for layer_index in find:
            for concept in find[layer_index]:
                new_trying = []
                trying = sorted(find[layer_index][concept].items(), key=lambda x: len(x[0]))
                for index, i in enumerate(trying):
                    if len(i[0]) < 2:
                        continue
                    length = len(i[0])
                    mark = True
                    for j in trying[index:]:
                        if len(j[0]) <= length:
                            continue
                        if i[0] in j[0]:
                            mark = False
                            break
                    if mark:
                        new_trying.append([i[0],i[1]])
                find[layer_index][concept] = new_trying
        new_find = {}
        for la in find:
            for key in find[la]:
                new_find[key] = find[la][key]
        self.query2spu_dishes_path = os.path.join(self.data_path,'spu自底向上撞推荐菜_最长字符串匹配.json')
        gzh.toJson(new_find, self.query2spu_dishes_path)

if __name__ == '__main__':
    import time
    start = time.time()
    mt = gzh.dr('../data')
    a = query2Dishes(10,'../data',mt.hypo_path,mt.poi_path,mt.spu_json,mt.dish_txt,mt.cate_txt,mt.poi_spu_filter,need_init_poi=False)
    print(list(a.query2fgc.keys()))
    txt = a.query2spu_dishes('葡萄')
    print(txt)
    '''txt = sorted(txt,key=lambda x:x[1],reverse=True)
    f = open('./aa.txt','w',encoding='utf-8')
    for i in txt:
        f.write(i[0]+'\n')
    f.close()
    print(a.query2poi_dishes(list(a.query2fgc.keys())[0],a=10,c=3))
    print(time.time()-start)'''

