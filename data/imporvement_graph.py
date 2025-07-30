import graphviz
import json
import re
import os


def split_long_str(string, every=5):
    lines = []
    for i in range(0, len(string), every):
        lines.append(string[i:i+every])
    return '\n'.join(lines)

class ImrovementGraph:

    def __init__(self,name='solution1_run1',file_dir='.') -> None:
        self.name = name
        self.file_dir = file_dir
        self.clear()
    
    def add_block(self,block_name,acc):
        if block_name in self.block2acc:
            raise NotImplementedError
        self.block2acc[block_name] = acc

    def add_edge(self,block_from,block_to,proposal):
        if block_from not in self.block2acc or block_to not in self.block2acc:
            raise NotImplementedError
        self.edges.append([block_from,block_to,proposal])

    def draw_graph(self,save_name=None):
        graph = graphviz.Digraph(self.name, format='png')
        graph.attr(rankdir='LR')
        for node,val in self.block2acc.items():
            graph.node(node,val,shape='circle')
        for edge in self.edges:
            graph.edge(edge[0],edge[1],label=edge[2].replace('_','\n'))
        if save_name==None:
            save_name = self.name
        graph.render(save_name,directory=self.file_dir, format='png', view=False,cleanup=True)

    def clear(self):
        self.block2acc = {}
        self.edges = []

    def txt2graph(self,txt):
        pattern1 = '(.*):(.*)'
        pattern2 = '(.*)--(.*)-->(.*)'
        lines = txt.strip().split('\n')
        for line in lines:
            line = line.strip()
            matches = re.findall(pattern1,line)
            if len(matches)!=0:
                match = matches[0]
                self.add_block(match[0],match[1])
            else:
                matches = re.findall(pattern2,line)
                if len(matches)==0:
                    raise NotImplementedError
                match = matches[0]
                self.add_edge(match[0],match[2],match[1])


if __name__=='__main__':
    txt = """block_1:77.32
block_2:77.62
block_3:76.50
block_4:74.3
block_1--squeeze_and_excitation-->block_2
block_1--depthwise_separable_convolution-->block_3
block_1--grouped_convolution-->block_4
"""
    graph = ImrovementGraph('solution1_run1')
    graph.txt2graph(txt)
    graph.draw_graph()