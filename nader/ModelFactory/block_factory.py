import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from torch.nn import (
    Conv2d,Linear,MaxPool2d,AvgPool2d,AdaptiveAvgPool2d,AdaptiveMaxPool2d,
    ReLU,GELU,Sigmoid,BatchNorm2d,LayerNorm
)
import networkx as nx
import numpy as np
import pandas as pd
import graphviz
import pickle
import json
import copy
from thop import profile
import logging
import pdb
import re

from ModelFactory.register import Registers, import_one_modules_for_register, import_module_from_path

class DAGError(Exception):
    def __init__(self, message="This is a custom error."):
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message

class BlockFactory():

    ops = ['Conv2d','Linear','AvgPool2d','MaxPool2d','AdaptiveMaxPool2d','AdaptiveAvgPool2d','Upsample',
            'ReLU','Sigmoid','GELU','BN','LN','mean','sum','max','permute','repeat','reshape',
            'concat','Add','Mul','multiply','softmax','DropPath']

    def __init__(self,
                blocks_dir=None,
                type='base',
                register_path='ModelFactory',
                stem_down_scale=4,
                mode='nas-bench-201') -> None:
        self.type = type
        self.register_path = register_path
        self.stem_down_scale = stem_down_scale
        self.mode = mode
        self.package = f'blocks.{type}.code'
        if blocks_dir==None:
            return
        self.anno_path = os.path.join(blocks_dir,'anno.json')
        self.code_dir = os.path.join(blocks_dir,'code')
        self.image_dir = os.path.join(blocks_dir,'image')
        self.txt_dir = os.path.join(blocks_dir,'txt')
        self.dag_dir = os.path.join(blocks_dir,'dag')
        self.dag_nx_dir = os.path.join(blocks_dir,'dag_nx')
        os.makedirs(self.code_dir,exist_ok=True)
        os.makedirs(self.image_dir,exist_ok=True)
        os.makedirs(self.txt_dir,exist_ok=True)
        os.makedirs(self.dag_dir,exist_ok=True)
        os.makedirs(self.dag_nx_dir,exist_ok=True)
        self.annos = self.load_anno()

    def __len__(self):
        return len(self.load_anno())

    def load_anno(self):
        if not os.path.isfile(self.anno_path):
            self.annos = {}
        else:
            with open(self.anno_path,'r') as f:
                self.annos = json.load(f)
        return self.annos

    def save_anno(self):
        with open(self.anno_path,'w') as f:
            json.dump(self.annos,f,indent='\t')
        # self.annos.to_csv(self.anno_path,index=False)

    def load_txt(self,path):
        with open(path,'r') as f:
            s = f.read()
        return s

    def split_sections(self, txt):
        """
        Split a multi-section BDAG text into ordered list of (section_name, section_text).
        A section starts with a line like: ##SectionName## and continues until next ##...## or EOF.
        """
        pattern = r'(##(.*?)##(.(?!##))*)'
        matches = re.findall(pattern, txt, flags=re.MULTILINE|re.DOTALL)
        sections = []
        for full, name, _ in matches:
            sections.append((name.strip(), full.strip('\n')))
        return sections

    def sanitize_id(self, name):
        """Sanitize a section name to be used as part of block id."""
        s = re.sub(r'[^A-Za-z0-9_]+','_', name)
        s = re.sub(r'_+','_', s).strip('_')
        return s

    def add_blocks_from_sections_path(self, path, id_prefix='fpn', with_isomorphic=True):
        """
        Detection-oriented helper: read a multi-section txt and create a block per section.
        Returns a dict mapping section_name -> block_id (or {'error': ...} on first failure).
        
        Special handling for FPN minimal mode:
        - If only 1 section (FPN_Lateral_base), auto-expand to 10 blocks
        - Lateral: 4 copies (P2, P3, P4, P5)
        - Output: 4 copies with kernel_size=3 (P2, P3, P4, P5)
        - Extra: Skip (handled in code)
        """
        import json
        
        txt = self.load_txt(path)
        sections = self.split_sections(txt)
        if len(sections) == 0:
            return {'error': 'No sections found in txt.'}
        
        # Check if this is FPN minimal mode (1 Lateral_base section only)
        is_fpn_minimal = (len(sections) == 1 and 
                         'FPN_Lateral_base' in sections[0][0])
        
        if is_fpn_minimal and self.mode == 'detection':
            print(f"[FPN Minimal Mode] Detected - Auto-expanding 1 block to 8 blocks")
            name2id = self._expand_fpn_minimal(sections[0][1], id_prefix, with_isomorphic)
            if isinstance(name2id, dict) and 'error' in name2id:
                return name2id
        else:
            # Standard mode: process each section as-is
            name2id = {}
            for section_name, section_txt in sections:
                sid = f"{id_prefix}__{self.sanitize_id(section_name)}"
                check_res = self.check(section_txt,with_isomorphic=with_isomorphic)
                if isinstance(check_res,dict):
                    return {'error': f"Section {section_name} invalid: {check_res['error']}"}
                add_res = self.add_block(section_txt, sid)
                if isinstance(add_res,dict):
                    return {'error': f"Section {section_name} codegen failed: {add_res['error']}"}
                name2id[section_name] = add_res
        
        # Save mapping to JSON file for the adapter to use
        mapping_file = os.path.join(os.path.dirname(self.anno_path), 'block_mapping.json')
        with open(mapping_file, 'w') as f:
            json.dump(name2id, f, indent=2, ensure_ascii=False)
        
        return name2id
    
    def _expand_fpn_minimal(self, lateral_base_txt, id_prefix, with_isomorphic):
        """
        Expand 1 FPN_Lateral_base block to 8 blocks:
        - 4 Lateral blocks (P2, P3, P4, P5) - same as base
        - 4 Output blocks (P2, P3, P4, P5) - kernel_size=1 -> kernel_size=3, padding=1
        """
        import re
        name2id = {}
        
        # 1. Create 4 Lateral blocks (identical to base)
        for level in ['P2', 'P3', 'P4', 'P5']:
            section_name = f'FPN_Lateral_{level}'
            sid = f"{id_prefix}__FPN_Lateral_{level}_base"
            
            check_res = self.check(lateral_base_txt, with_isomorphic=with_isomorphic)
            if isinstance(check_res, dict):
                return {'error': f"Lateral {level} invalid: {check_res['error']}"}
            
            add_res = self.add_block(lateral_base_txt, sid)
            if isinstance(add_res, dict):
                return {'error': f"Lateral {level} codegen failed: {add_res['error']}"}
            
            name2id[section_name] = add_res
            print(f"  → Created FPN_Lateral_{level}_base")
        
        # 2. Create 4 Output blocks (kernel_size=1 -> 3, add padding=1)
        output_txt = lateral_base_txt
        # Replace kernel_size=1 with kernel_size=3
        output_txt = re.sub(r'kernel_size=1', 'kernel_size=3', output_txt)
        # Add padding=1 if not present
        if 'padding=' not in output_txt:
            output_txt = re.sub(
                r'(Conv2d\([^)]*stride=1)',
                r'\1,padding=1',
                output_txt
            )
        
        for level in ['P2', 'P3', 'P4', 'P5']:
            section_name = f'FPN_Output_{level}'
            sid = f"{id_prefix}__FPN_Output_{level}_base"
            
            check_res = self.check(output_txt, with_isomorphic=with_isomorphic)
            if isinstance(check_res, dict):
                return {'error': f"Output {level} invalid: {check_res['error']}"}
            
            add_res = self.add_block(output_txt, sid)
            if isinstance(add_res, dict):
                return {'error': f"Output {level} codegen failed: {add_res['error']}"}
            
            name2id[section_name] = add_res
            print(f"  → Created FPN_Output_{level}_base")
        
        print(f"[FPN Minimal Mode] Successfully expanded to {len(name2id)} blocks")
        return name2id

    def save_txt(self,txt,path):
        with open(path,'w') as f:
            f.write(txt)

    def load_dag(self,path):
        with open(path,'r') as f:
            ds = json.load(f)
        return ds
    
    def save_dag(self,g,path):
        with open(path,'w') as f:
            json.dump(g,f,indent='\t')

    def load_dag_nx(self,path):
        with open(path, 'rb') as f:
            g = pickle.load(f)
        return g

    def save_dag_nx(self,g,path):
        with open(path, 'wb') as f:
            pickle.dump(g, f, pickle.HIGHEST_PROTOCOL)

    def get_block_txt(self,block_id):
        with open(os.path.join(self.txt_dir,f"{block_id}.txt"),'r') as f:
            txt = f.read()
        return txt
        
    def parse_txt(self, s, get_shape=True):
        nodes = {}
        edges = []
        s = s.strip('\n')
        ls = re.split('\n+',s)
        name = ls[0].replace('#','')
        
        # ===== COMMON STRICT VALIDATION: Check block name (all modes except darts) =====
        if self.mode != 'darts':
            if not name or len(name.strip()) == 0:
                raise DAGError('Block name cannot be empty.')
            if len(name) > 100:
                raise DAGError(f'Block name too long ({len(name)} chars). Maximum 100 characters allowed.')
        
        for l in ls[1:]:
            l = l.split('#')[0].strip()
            if ':' in l:
                its = l.split(':')
                if len(its) != 2:
                    raise DAGError(f'Invalid node definition: {l}. Expected format: "node_id:operation"')
                
                node_id = its[0].strip()
                operation = its[1].strip()
                
                # ===== COMMON STRICT VALIDATION: Check node ID (all modes except darts) =====
                if self.mode != 'darts':
                    if not node_id:
                        raise DAGError('Node ID cannot be empty.')
                    # Node ID should be numeric or valid identifier
                    if not (node_id.isdigit() or node_id.replace('_','').isalnum()):
                        raise DAGError(f'Invalid node ID: {node_id}. Node IDs should be numeric or alphanumeric with underscores.')
                    
                    # Check operation is not empty
                    if not operation:
                        raise DAGError(f'Node {node_id} has empty operation.')
                
                if node_id not in nodes:
                    nodes[node_id] = operation
                else:
                    raise DAGError(f'node {node_id} error: Duplicate node name error.')
            elif '->' in l:
                its = l.split('->')
                if len(its) != 2:
                    raise DAGError(f'Invalid edge definition: {l}. Expected format: "source->target" or "src1,src2->target"')
                
                source = its[0].strip()
                target = its[1].strip()
                
                # ===== COMMON STRICT VALIDATION: Check edge format (all modes except darts) =====
                if self.mode != 'darts':
                    if not source or not target:
                        raise DAGError(f'Invalid edge: {l}. Source and target cannot be empty.')
                
                if ',' in source:
                    for i in source.split(','):
                        i = i.strip()
                        if not i and self.mode != 'darts':
                            raise DAGError(f'Invalid edge source in multi-source edge: {l}')
                        edges.append([i, target])
                else:
                    edges.append([source, target])
            else:
                continue
        
        # ===== COMMON STRICT VALIDATION: Check minimum requirements (all modes except darts) =====
        if self.mode != 'darts':
            if len(nodes) < 3:
                raise DAGError(f'Block must have at least 3 nodes (input, operation, output), got {len(nodes)}.')
            if len(edges) < 2:
                raise DAGError(f'Block must have at least 2 edges, got {len(edges)}.')
            
            # Check for input and output nodes
            has_input = False
            has_output = False
            for node_id, operation in nodes.items():
                op_lower = operation.lower()
                if self.mode == 'detection':
                    if op_lower.startswith('input'):
                        has_input = True
                else:
                    # NAS-Bench mode: exact match "input"
                    if op_lower == 'input':
                        has_input = True
                if op_lower == 'output':
                    has_output = True
            
            if not has_input:
                if self.mode == 'detection':
                    raise DAGError('Block must have at least one input node (operation starting with "input").')
                else:
                    raise DAGError('Block must have exactly one input node (operation "input").')
            if not has_output:
                raise DAGError('Block must have exactly one output node (operation "output").')
        
        # check all nodes are defined
        for edge in edges:
            for node in edge:
                if node not in nodes:
                    raise DAGError(f'edge {edge[0]}->{edge[1]} error: Node {node} is not defined.')
        
        # ===== COMMON STRICT VALIDATION: Check for self-loops and duplicate edges (all modes except darts) =====
        if self.mode != 'darts':
            for edge in edges:
                if edge[0] == edge[1]:
                    raise DAGError(f'Self-loop detected: {edge[0]}->{edge[1]}. Self-loops are not allowed.')
            
            # Check for duplicate edges
            edge_set = set()
            for edge in edges:
                edge_tuple = (edge[0], edge[1])
                if edge_tuple in edge_set:
                    raise DAGError(f'Duplicate edge detected: {edge[0]}->{edge[1]}.')
                edge_set.add(edge_tuple)
        
        dag = {
            'name':name,
            'nodes':nodes,
            'edges':edges
        }
        if get_shape:
            dag['node2shape'],dag['node2output'] = self.get_shape(dag)
        return dag
    
    def drawDAG(self,dag,edge_label='node2output',draw_edge_label=False,file_name=None,file_dir='.'):
        if isinstance(dag,str):
            dag = self.parse_txt(dag)
        dot = graphviz.Digraph(dag['name'], format='png', graph_attr={'rankdir': 'TB'})
        for node,val in dag['nodes'].items():
            if val.lower() in ['input','output']:
                dot.node(node,val,shape='ellipse')
            else:
                dot.node(node,val,shape='Mrecord')
        for edge in dag['edges']:
            if draw_edge_label and edge_label in dag:
                dot.edge(edge[0],edge[1],label=str(tuple(dag['node2output'][edge[0]])))
                dot.attr(label=f"Params:{dag['params']}M\nFLOPs:{dag['flops']}G",labelloc='tl',labeljust='l')
            else:
                dot.edge(edge[0],edge[1])
        if file_name==None:
            file_name = dag['name']
        dot.render(file_name,directory=file_dir, format='png', view=False,cleanup=True)

    def split_op(self,node,op,with_check=True):
        def replace_div(s):
            if isinstance(s,str):
                if '/' in s and '//' not in s:
                    s = s.replace('/','//')
            return s
        op = op.replace('[','(')
        op = op.replace(']',')')
        op1 = op.split('(')
        if len(op1)==1:
            return op1[0], None
        op = re.findall(r'([^(]*)\((.*)\)', op)[0]
        op,param=op
        param = re.split(r',(?![^\(\)]*\))', param)
        param = [p.strip() for p in param]
        assert len(param)>0
        if '=' not in param[0]:
            # permute, repeat, reshape
            if op in ['reshape']:
                if param[0] not in ['B','-1']:
                    raise DAGError(f"node {node} error: {op} operation's first dimension must be B.")
            return op,param
        params = {}
        for p in param:
            key,val = p.split('=')
            if ',' in val:
                val = val.strip('(').strip(')').split(',')
                val = [v.strip() for v in val]
            else:
                val = val.strip()
            # normalize quoted strings like 'nearest' or "nearest" → nearest
            def _strip_quotes(x):
                if isinstance(x,str) and len(x)>=2 and ((x[0]==x[-1]=="'") or (x[0]==x[-1]=='"')):
                    return x[1:-1]
                return x
            if isinstance(val,list):
                val = [_strip_quotes(v) for v in val]
            else:
                val = _strip_quotes(val)
            params[key.strip()] = val
        if op=='Conv2d':
            if 'out_channels' not in params:
                raise DAGError(f'node {node} error: {op} operation must has out_channels parameter.')
            if 'kernel_size' not in params:
                raise DAGError(f'node {node} error: {op} operation must has kernel_size parameter.')
            params['out_channels'] = replace_div(params['out_channels'])
            params['stride'] = replace_div(params.get('stride',1))
            params['dilation'] = replace_div(params.get('dilation',1))
            params['groups'] = replace_div(params.get('groups',1))
            
            # ===== COMMON STRICT VALIDATION: Conv2d parameters (all modes except darts) =====
            if self.mode != 'darts':
                # Validate kernel_size (should be reasonable, e.g., 1-11)
                try:
                    ks = params['kernel_size']
                    if isinstance(ks, str) and ks.isdigit():
                        ks_val = int(ks)
                        if ks_val < 1 or ks_val > 11:
                            raise DAGError(f'node {node} error: Conv2d kernel_size must be between 1 and 11, got {ks_val}.')
                except:
                    pass  # Skip validation for symbolic values like "C"
                
                # Validate stride
                try:
                    stride = params['stride']
                    if isinstance(stride, str) and stride.isdigit():
                        stride_val = int(stride)
                        if stride_val < 1 or stride_val > 4:
                            raise DAGError(f'node {node} error: Conv2d stride must be between 1 and 4, got {stride_val}.')
                except:
                    pass
                
                # Validate dilation
                try:
                    dilation = params['dilation']
                    if isinstance(dilation, str) and dilation.isdigit():
                        dilation_val = int(dilation)
                        if dilation_val < 1 or dilation_val > 4:
                            raise DAGError(f'node {node} error: Conv2d dilation must be between 1 and 4, got {dilation_val}.')
                except:
                    pass
            # if params['groups'] != 1:
            #     params['out_channels'] = f"{params['out_channels']}//{params['groups']}*{params['group']}"
        elif op=='Linear':
            if 'out_channels' not in params:
                raise DAGError(f'node {node} error: {op} operation must has out_channels parameter.')
            params['out_channels'] = replace_div(params['out_channels'])
        elif op in ['AvgPool2d','MaxPool2d']:
            if 'stride' not in params:
                raise DAGError(f'node {node} error: {op} operation must has stride parameter.')
            if 'kernel_size' not in params:
                raise DAGError(f'operation {node} error: {op} operation must has kernel_size parameter.')
            params['kernel_size'] = replace_div(params['kernel_size'])
            params['stride'] = replace_div(params['stride'])
            
            # ===== COMMON STRICT VALIDATION: Pooling parameters (all modes except darts) =====
            if self.mode != 'darts':
                try:
                    ks = params['kernel_size']
                    if isinstance(ks, str) and ks.isdigit():
                        ks_val = int(ks)
                        if ks_val < 1 or ks_val > 8:
                            raise DAGError(f'node {node} error: {op} kernel_size must be between 1 and 8, got {ks_val}.')
                except:
                    pass
                
                try:
                    stride = params['stride']
                    if isinstance(stride, str) and stride.isdigit():
                        stride_val = int(stride)
                        if stride_val < 1 or stride_val > 8:
                            raise DAGError(f'node {node} error: {op} stride must be between 1 and 8, got {stride_val}.')
                except:
                    pass
        elif op in ['AdaptiveMaxPool2d','AdaptiveAvgPool2d']:
            if 'output_size' not in params:
                raise DAGError(f'node {node} error: {op} operation must has output_size parameter.')
            params['output_size'] = replace_div(params['output_size'])
        elif op in ['concat','mean','max','sum','softmax']:
            if 'dim' not in params:
                raise DAGError(f'node {node} error: {op} operation must has dim parameter.')
            
            # ===== COMMON STRICT VALIDATION: Dimension parameter (all modes except darts) =====
            if self.mode != 'darts':
                try:
                    dim = params['dim']
                    if isinstance(dim, str) and (dim.lstrip('-').isdigit()):
                        dim_val = int(dim)
                        if abs(dim_val) > 4:
                            raise DAGError(f'node {node} error: {op} dim must be between -4 and 4, got {dim_val}.')
                except:
                    pass
        elif op in ['Upsample']:
            # support either scale_factor or size, and optional mode
            if 'scale_factor' not in params and 'size' not in params:
                raise DAGError(f'node {node} error: {op} operation must have scale_factor or size parameter.')
            # normalize mode default
            if 'mode' not in params:
                params['mode'] = 'nearest'
            
            # ===== COMMON STRICT VALIDATION: Upsample parameters (all modes except darts) =====
            if self.mode != 'darts':
                if 'scale_factor' in params:
                    try:
                        sf = params['scale_factor']
                        if isinstance(sf, str):
                            sf_val = float(sf)
                            if sf_val < 0.5 or sf_val > 8.0:
                                raise DAGError(f'node {node} error: Upsample scale_factor must be between 0.5 and 8.0, got {sf_val}.')
                    except ValueError:
                        pass
                
                # Validate mode
                if 'mode' in params:
                    valid_modes = ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']
                    if params['mode'] not in valid_modes:
                        raise DAGError(f'node {node} error: Upsample mode must be one of {valid_modes}, got {params["mode"]}.')
        # elif op in ['DropPath']:
        #     if 'prob' not in params:
        #         raise DAGError(f'node {node} error: {op} operation must has prob parameter.')
        return op,params
    
    def dag2nx(self,dag,with_check=True):
        g = nx.DiGraph()
        g.add_edges_from(dag['edges'])
        for node in dag['nodes']:
            if node not in g.nodes:
                if with_check:
                    raise DAGError(f"node {node} error: The node is not used.")
                else:
                    g.add_node(node)
        node_values = {}
        node_ops = {}
        node_params = {}
        for i,val in dag['nodes'].items():
            assert i not in node_values
            node_values[i] = val
            if with_check:
                op,params = self.split_op(i,val)
                node_ops[i] = op
                node_params[i] = params
            else:
                node_ops[i] = val
                node_params[i] = None
        nx.set_node_attributes(g,node_values,'value')
        nx.set_node_attributes(g,node_ops,'op')
        nx.set_node_attributes(g,node_params,'params')
        return g
    
    def shape2val(self,shapes,B=4,C=64,H=32,W=32,dim=32):
        if self.mode=='darts' or self.type=='base':
            C = dim
        shapes = copy.deepcopy(shapes)
        if isinstance(shapes,(list,tuple)):
            for i, shape in enumerate(shapes):
                shapes[i] = self.shape2val(shape)
        if isinstance(shapes,str):
            shapes = eval(shapes)
        if isinstance(shapes,float):
            shapes = int(shapes)
        return shapes

    def get_shape(self,dag,cB=4,cC=64,cH=32,cW=32,cdim=32):
        node2shape = {}
        node2out = {}
        g = nx.DiGraph()
        g.add_edges_from(dag['edges'])
        for node in list(nx.topological_sort(g)):
            op,params = self.split_op(node,dag['nodes'][node])
            ns = list(g.predecessors(node))
            # accept input nodes like input, input1, input2, input_Pk
            if op.lower() in ['input','input1','input2'] or op.lower().startswith('input'):
                # detection: some inputs (lateral/merged/output) are already in dim channels
                if self.mode=='detection':
                    val_str = str(dag['nodes'][node]).lower()
                    if val_str.startswith('input_lat_'):
                        node2shape[node] = [['B','dim','H','W'],['B','dim','H','W']]
                    elif val_str.startswith('input_merged_'):
                        # Decide spatial size by level: input_merged_Pn feeding block *_Pn keeps H,W; feeding *_P(n-1) is smaller.
                        import re as _re
                        def _get_p(s):
                            m = _re.search(r'_p(\d+)', s)
                            return int(m.group(1)) if m else None
                        lvl_in = _get_p(val_str)
                        lvl_blk = _get_p(dag['name'].lower())
                        if lvl_in is not None and lvl_blk is not None and lvl_in == lvl_blk:
                            node2shape[node] = [['B','dim','H','W'],['B','dim','H','W']]
                        else:
                            node2shape[node] = [['B','dim','H//2','W//2'],['B','dim','H//2','W//2']]
                    elif val_str.startswith('input_output_'):
                        node2shape[node] = [['B','dim','H','W'],['B','dim','H','W']]
                    elif val_str.startswith('input_c'):
                        node2shape[node] = [['B','C','H','W'],['B','C','H','W']]
                    else:
                        node2shape[node] = [['B','C','H','W'],['B','C','H','W']]
                else:
                    node2shape[node] = [['B','C','H','W'],['B','C','H','W']]
                out_shapes_v = self.shape2val(node2shape[node][1])
                node2out[node] = out_shapes_v
                continue
            elif op.lower()=='output':
                if len(ns)!=1:
                    raise DAGError(f'node {node} error: The output node must has only one input.')
                node2shape[node] = [copy.deepcopy(node2shape[ns[0]][1]),['B','dim','H','W']]
                in_shapes_v = self.shape2val(node2shape[node][0])
                if self.type in ['base','normal']:
                    if self.mode=='detection':
                        # detection: only enforce channel==dim; spatial size can differ (e.g., stride-2 extra conv)
                        if in_shapes_v[1]!=cdim:
                            raise DAGError(f'node {node} error: Output channel must be dim.')
                    else:
                        if in_shapes_v[1]!=cdim or in_shapes_v[2]!=cH or in_shapes_v[3]!=cW:
                            raise DAGError(f'node {node} error: Output shape must be (B,dim,H,W).')
                elif self.type in ['stem','downsample']:
                    if self.mode=='nas-bench' and in_shapes_v[1]!=cdim:
                        raise DAGError(f"node {node} error: Output's channel dimension must be dim.")
                if self.type=='stem' and self.mode=='nas-bench':
                    if in_shapes_v[2]!=cH/self.stem_down_scale or in_shapes_v[3]!=cW/self.stem_down_scale:
                        raise DAGError(f'node {node} error: The height and width of the output feature map must be {1/self.stem_down_scale} of the input feature map.')
                elif self.type in ['downsample','reduction'] and self.mode=='nas-bench':
                    if in_shapes_v[2]!=cH/2 or in_shapes_v[3]!=cW/2:
                        raise DAGError(f'node {node} error: The height and width of the output feature map must be 1/2 of the input feature map.')
                out_shapes_v = self.shape2val(node2shape[node][1])
                node2out[node] = out_shapes_v
                continue
            if op not in self.ops and op.lower() not in self.ops:
                raise DAGError(f'node {node} error: Undefined computation {op} is used')
            if len(ns)==1:
                # Conv2d, Linear, AvgPool2d, MaxPool2d, AdaptiveMaxPool2d, AdaptiveAvgPool2d
                # ReLU, Sigmoid, GELU
                # BN, LN
                # DropPath
                # mean,sum,max
                # permute,repeat,reshape
                if op in ['Conv2d','Linear','AvgPool2d','MaxPool2d','AdaptiveMaxPool2d','AdaptiveAvgPool2d',
                            'ReLU','Sigmoid','GELU','BN','LN','DropPath']:
                    if params and 'out_channels' in  params:
                        dim = params['out_channels']
                    else:
                        dim = copy.deepcopy(node2shape[ns[0]][1][1])
                    if params:
                        for key,val in params.items():
                            if 'H' in str(val) or 'W' in str(val):
                                raise DAGError("node {node} error: H and W cannot appear in layer definition.")
                    if params and 'stride' in params:
                        spa_scale = params['stride']
                    else:
                        spa_scale = '1'
                    input_shapes = copy.deepcopy(node2shape[ns[0]][1])
                    out_shapes = copy.deepcopy(input_shapes)
                    input_shapes_v = node2out[ns[0]]
                    out_shapes_v = copy.deepcopy(input_shapes_v)
                    if op in ['Linear']:
                        out_shapes[-1] = dim
                    elif op in ['AdaptiveMaxPool2d','AdaptiveAvgPool2d']:
                        size = params['output_size']
                        if ',' not in size:
                            size = int(size)
                            out_shapes[2] = size
                            out_shapes[3] = size
                        else:
                            size = [int(s.strip()) for s in size.strip('(').strip(')').split(',')]
                            assert len(size)==2
                            out_shapes[2] = size[0]
                            out_shapes[3] = size[1]
                        out_shapes[1] = dim
                    else:
                        out_shapes[1] = dim
                    for i in range(2,len(out_shapes)):
                        out_shapes[i] = f'({out_shapes[i]})//({spa_scale})'
                    if op in ['Linear','LN']:
                        if 'W' in input_shapes[-1] or 'W' in out_shapes[-1]:
                            raise DAGError(f'node {node} error: When using {op}, the last dimension must be the channel dimension.')
                    node2shape[node] = [input_shapes,out_shapes]
                    # process -1
                    for i in range(len(out_shapes)):
                        if input_shapes[i]!=out_shapes[i]:
                            out_shapes_v[i]=out_shapes[i]
                    out_shapes_v = self.shape2val(out_shapes_v)
                elif op in ['Upsample']:
                    # handle upsample by scale_factor or size
                    input_shapes = copy.deepcopy(node2shape[ns[0]][1])
                    out_shapes = copy.deepcopy(input_shapes)
                    if params and 'size' in params:
                        size = params['size']
                        if isinstance(size,list):
                            assert len(size)==2
                            out_shapes[2] = size[0]
                            out_shapes[3] = size[1]
                        else:
                            # single int means square size
                            out_shapes[2] = size
                            out_shapes[3] = size
                    else:
                        # default by scale_factor
                        sf = params.get('scale_factor','2')
                        out_shapes[2] = f'({out_shapes[2]})*({sf})'
                        out_shapes[3] = f'({out_shapes[3]})*({sf})'
                    node2shape[node] = [input_shapes,out_shapes]
                    out_shapes_v = self.shape2val(out_shapes)
                elif op in ['mean','sum','max']:
                    assert 'dim' in params
                    out_shapes = copy.deepcopy(node2shape[ns[0]][1])
                    if isinstance(params['dim'],list):
                        for dim in params['dim']:
                            out_shapes[int(dim)] = 1
                    else:
                        out_shapes[int(params['dim'])] = 1
                    node2shape[node] = [node2shape[ns[0]][1],out_shapes]
                    out_shapes_v = self.shape2val(out_shapes)
                elif op.lower() in ['softmax']:
                    assert 'dim' in params
                    out_shapes = node2shape[ns[0]][1]
                    node2shape[node] = [node2shape[ns[0]][1],out_shapes]
                    out_shapes_v = self.shape2val(out_shapes)
                elif op == 'permute':
                    dims = self.shape2val(params)
                    if len(dims)==len(node2shape[ns[0]][1]):
                        out_shapes = []
                        for dim in dims:
                            out_shapes.append(node2shape[ns[0]][1][int(dim)])
                        node2shape[node] = [node2shape[ns[0]][1],out_shapes]
                    else:
                        raise DAGError(f'node {node} error: {op} dims length error')
                    out_shapes_v = self.shape2val(out_shapes)
                elif op == 'repeat':
                    input_shapes = copy.deepcopy(node2shape[ns[0]][1])
                    try:
                        sizes = [eval(param) for param in params]
                    except:
                        raise DAGError('node {node} error: repeat sizes must numbers')
                    if len(sizes) >= len(input_shapes):
                        if len(sizes) > len(input_shapes):
                            input_shapes = ['1']*(len(input_shapes)-len(sizes)) + input_shapes
                        out_shapes = []
                        for s1,s2 in zip(input_shapes,sizes):
                            out_shapes.append(f'({s1})*({str(s2)})')
                        node2shape[node] = [input_shapes,out_shapes]
                    else:
                        raise DAGError(f"node {node} error: {op} operation's sizes length must be same the length of its input's shape.")
                    out_shapes_v = self.shape2val(out_shapes)
                elif op=='reshape':
                    shapes = self.shape2val(params)
                    if len(params)<=1:
                        raise DAGError(f"node {node} error: {op} operation must has at least one dim.")
                    node2shape[node] = [node2shape[ns[0]][1],params]
                    in_shapes_v = node2out[ns[0]]
                    out_shapes_v = copy.deepcopy(shapes)
                    shapes_v = [in_shapes_v,out_shapes_v]
                    s1,s2,flag=1,1,-1
                    for shape in shapes_v[0]:
                        s1*=shape
                    for i,shape in enumerate(shapes_v[1]):
                        if shape==-1:
                            if flag!=-1:
                                raise DAGError(f"node {node} error: {op} operation's dim error")
                            else:
                                flag = i
                        else:
                            s2*=shape
                    if s1%s2!=0:
                        raise DAGError(f"node {node} error: {op} operation's dim error")
                    if flag!=-1:
                        shapes_v[1][flag]=int(s1/s2)
                    out_shapes_v = shapes_v[1]
                else:
                    raise DAGError(f'node {node} error: {op} cannot receive only one input.')
            elif len(ns)>1:
                if op.lower()=='concat':
                    assert 'dim' in params
                    dim = int(params['dim'])
                    input_shapes = [copy.deepcopy(node2shape[ns[0]][1])]
                    c = node2shape[ns[0]][1][dim]
                    for i in range(1,len(ns)):
                        input_shapes.append(node2shape[ns[i]][1])
                        c = f"({c})+({node2shape[ns[i]][1][dim]})"
                    out_shapes = copy.deepcopy(node2shape[ns[0]][1])
                    out_shapes[dim] = c
                    node2shape[node] = [input_shapes,out_shapes]
                    shapes_v = self.shape2val(node2shape[node])
                    dim_len = 0
                    for i in range(len(shapes_v[0])):
                        if len(shapes_v[0][i]) != len(shapes_v[1]):
                            raise DAGError(f"node {node} error: {op} operation's inputs can not be {op}ed.")
                        for j in range(len(shapes_v[0][i])):
                            if j!=dim and shapes_v[0][i][j]!=shapes_v[1][j]:
                                raise DAGError(f"node {node} error: {op} operation's inputs can not be {op}ed.")
                            if j==dim:
                                dim_len+=shapes_v[0][i][j]
                    if dim_len!=shapes_v[1][dim]:
                        raise DAGError(f"node {node} error: {op} operation's inputs can not be {op}ed.")
                    out_shapes_v = shapes_v[1]
                elif op.lower() in ['add','mul']:
                    input_shapes,input_shapes_v = [],[]
                    for i in range(len(ns)):
                        input_shapes.append(copy.deepcopy(node2shape[ns[i]][1]))
                        input_shapes_v.append(copy.deepcopy(node2out[ns[i]]))
                    out_shapes = ['None']*len(input_shapes)
                    max_len = 0
                    for shape in input_shapes_v:
                        if isinstance(shape,int):
                            if max_len<1:
                                max_len=1
                        elif isinstance(shape,(tuple,list)):
                            s = len(shape)
                            if max_len<s:
                                max_len=s
                    out_shapes = ['1']*max_len
                    out_shapes_v = [1]*max_len
                    for shape,shape_v in zip(input_shapes,input_shapes_v):
                        for i in range(-1,-1-len(shape),-1):
                            if shape_v[i]==1 or shape_v[i]==out_shapes_v[i]:
                                continue
                            if out_shapes_v[i]!=1:
                                raise DAGError(f"node {node} error: {op} operation's inputs can not be {op}ed.")
                            out_shapes[i] = shape[i]
                            out_shapes_v[i] = shape_v[i]
                    node2shape[node] = [input_shapes,out_shapes]
                elif op.lower()=='multiply':
                    if len(ns)!=2:
                        raise DAGError(f'node {node} error: {op} operation must have two inputs.')
                    x_shapes,y_shapes = copy.deepcopy(node2shape[ns[0]][1]),copy.deepcopy(node2shape[ns[1]][1])
                    x_shapes_v = self.shape2val(x_shapes)
                    y_shapes_v = self.shape2val(y_shapes)
                    input_shapes = [x_shapes,y_shapes]
                    if x_shapes_v[-1]==y_shapes_v[-2]:
                        out_shapes_v = copy.deepcopy(x_shapes_v)
                        out_shapes_v[-1] = y_shapes_v[-1]
                        out_shapes = copy.deepcopy(x_shapes)
                        out_shapes[-1] = y_shapes[-1]
                    elif x_shapes_v[-2]==y_shapes_v[-1]:
                        out_shapes_v = copy.deepcopy(y_shapes_v)
                        out_shapes_v[-1] = x_shapes_v[-1]
                        out_shapes = copy.deepcopy(y_shapes)
                        out_shapes[-1] = x_shapes[-1]
                    else:
                        raise DAGError(f"node {node} error: {op} operation's inputs can not be multiplied.")
                    node2shape[node] = [input_shapes,out_shapes]
                else:
                    raise DAGError(f'node {node} error: {op} operation can receive only one input.')
            node2out[node] = out_shapes_v
        return node2shape,node2out
    
    def check_isomorphic(self,dag_nx):
        self.load_anno()
        def cmp(n1,n2):
            if n1['op'].lower() != n2['op'].lower():
                return False
            if n1['params'] != n2['params']:
                return False
            return True
        if len(self.annos)==0:
            return -1
        for id in self.annos:
            block_nx_path = os.path.join(self.dag_nx_dir,f"{id}.gpickle")
            block = self.load_dag_nx(block_nx_path)
            if nx.is_isomorphic(dag_nx,block,node_match=cmp):
                return id
        return -1

    def check_two_isomorphic(self,dag1,dag2):
        if isinstance(dag1,str):
            dag1 = self.parse_txt(dag1)
            dag1 = self.dag2nx(dag1)
        if isinstance(dag2,str):
            dag2 = self.parse_txt(dag2)
            dag2 = self.dag2nx(dag2)
        def cmp(n1,n2):
            if n1['op'] != n2['op']:
                return False
            if n1['params'] != n2['params']:
                return False
            return True
        return nx.is_isomorphic(dag1,dag2,node_match=cmp)
        
    def dag2code(self,dag,block_name='Block',save_file_path='z.py'):
        g = nx.DiGraph()
        g.add_edges_from(dag['edges'])
        tab = '    '
        fun_init = ""
        fun_forward = ""
        # define node
        node2op = {}
        op_num = 1
        def replace(s, map={'C':'in_channels','dim':'out_channels'}):
            if isinstance(s,(tuple,list)):
                new_s = []
                for i in s:
                    new_s.append(replace(i))
            elif isinstance(s,str):
                new_s = s
                new_s = re.sub(r'(?<!/)/(?!/)', '//', new_s)
                for key in map:
                    new_s = new_s.replace(key,map[key])
                if not any((s.isalpha() for s in new_s)):
                    new_s = eval(new_s)
            else:
                new_s = int(s)
            return new_s
        node2shape = dag['node2shape']
        for i, node in dag['nodes'].items():
            op,params = self.split_op(i,node)
            if op == 'Conv2d':
                shapes = node2shape[i]
                in_channels = replace(shapes[0])[1]
                out_channels = replace(shapes[1])[1]
                if 'groups' in params:
                    groups = replace(params['groups'])
                else:
                    groups = 1
                kernel_size = replace(params['kernel_size'])
                if 'stride' in params:
                    stride = replace(params['stride'])
                else:
                    stride = 1
                if 'dilation' in params:
                    dilation = replace(params['dilation'])
                else:
                    dilation = 1
                row = f"CustomConv2d(in_channels={in_channels},out_channels={out_channels},kernel_size={replace(params['kernel_size'])},stride={stride},dilation={dilation},groups={groups})"
            elif op=='Linear':
                shapes = node2shape[i]
                in_channels = replace(shapes[0])[-1]
                out_channels = replace(shapes[1])[-1]
                row = f"Linear(in_features={in_channels},out_features={out_channels})"
            elif op in ['AvgPool2d','MaxPool2d']:
                kernel_size = replace(params['kernel_size'])
                stride = replace(params['stride'])
                row = f"Custom{op}(kernel_size={kernel_size},stride={stride})"
            elif op in ['AdaptiveAvgPool2d','AdaptiveMaxPool2d']:
                output_size = replace(params['output_size'])
                row = f"{op}(output_size={output_size})"
            elif op == 'BN':
                shapes = node2shape[i]
                out_channels = replace(shapes[1])[1]
                row = f"BatchNorm2d(num_features={out_channels})"
            elif op == 'LN':
                shapes = node2shape[i]
                out_channels = replace(shapes[1])[-1]
                row = f"LayerNorm(normalized_shape={out_channels})"
            elif op in ['Sigmoid','ReLU','GELU']:
                row = f"{op}()"
            elif op == 'Upsample':
                # build Upsample layer in __init__
                # prefer size if provided, otherwise scale_factor
                scale_factor = None
                size = None
                mode = 'nearest'
                if params:
                    if 'size' in params:
                        size = replace(params['size'])
                    if 'scale_factor' in params:
                        scale_factor = replace(params['scale_factor'])
                    if 'mode' in params:
                        mode = params['mode'] if isinstance(params['mode'],str) else str(params['mode'])
                if size is not None:
                    row = f"CustomUpsample(size={size}, mode={repr(mode)})"
                elif scale_factor is not None:
                    row = f"CustomUpsample(scale_factor={scale_factor}, mode={repr(mode)})"
                else:
                    row = f"CustomUpsample(scale_factor=2, mode={repr(mode)})"
            else:
                continue
            row = f"self.op{op_num}={row}"
            node2op[i] = f"self.op{op_num}"
            op_num+=1
            fun_init = fun_init + '\n' + tab*2+row
        # define compute graph
        node2out_var = {}
        var_num = 0
        for i in list(nx.topological_sort(g)):
            op,params = self.split_op(i,dag['nodes'][i])
            if op=='input' or op.lower().startswith('input'):
                node2out_var[i] = 'x'
                continue
            elif op=='input1':
                node2out_var[i] = 'x1'
                continue
            elif op=='input2':
                node2out_var[i] = 'x2'
                continue
            # in_var,out_var
            pres = list(g.predecessors(i))
            if len(pres)>1:
                in_var = [node2out_var[pre] for pre in pres]
                flag = True
                for pr in pres:
                    if len(list(g.successors(pres[0])))==1:
                        out_var = node2out_var[pr] 
                        flag = False
                        break
                if flag:
                    var_num += 1
                    out_var = f"h{var_num}"
            elif len(pres)==1:
                in_var = node2out_var[pres[0]]
                if len(list(g.successors(pres[0])))==1:
                    out_var = in_var
                else:
                    var_num += 1
                    out_var = f"h{var_num}"
            # op
            # print(op)
            if op=='output':
                assert len(pres)==1
                row = f"return {in_var}"
            if i in node2op:
                # Conv2d,Linear,AvgPool2d,MaxPool2d,AdaptiveAvgPool2d,AdaptiveMaxPool2d
                # BN,LN,Sigmoid,ReLU,GELU
                assert len(pres)==1
                row = f"{out_var} = {node2op[i]}({in_var})"
            elif op.lower() in ['concat']:
                # concat,mean,max,sum
                s = f"[{in_var[0]}"
                for x in in_var[1:]:
                    s=s+f",{x}"
                s = s+']'
                if isinstance(params['dim'],list):
                    dim = f"({','.join(params['dim'])})"
                else:
                    dim = int(params['dim'])
                row = f"{out_var} = torch.{op.lower()}({s}, dim={dim})"
            elif op.lower() in ['mean','max','sum','softmax']:
                # concat,mean,max,sum
                assert isinstance(in_var,str)
                if isinstance(params['dim'],list):
                    dim = f"({','.join(params['dim'])})"
                else:
                    dim = int(params['dim'])
                if op.lower() in ['softmax']:
                    row = f"{out_var} = torch.{op.lower()}({in_var}, dim={dim})"
                else:
                    row = f"{out_var} = torch.{op.lower()}({in_var}, dim={dim}, keepdim=True)"
            elif op.lower() in ['permute','repeat','reshape']:
                # permute,repeat,reshape
                p = params
                p = replace(p)
                row = f"{out_var} = {in_var}.{op}({p[0]}"
                for x in p[1:]:
                    row = row + f",{x}"
                row = row + ')'
            elif op == 'Upsample':
                # use F.interpolate or nn.Upsample; keep module style for consistency
                # define as layer during init already
                row = f"{out_var} = {node2op[i]}({in_var})"
            elif op.lower() in ['add','mul']:
                # Add, Mul
                op_map = {'add':'+','mul':'*'}
                row = f"{out_var} = {op_map[op.lower()].join(in_var)}"
            elif op.lower() == 'multiply':
                # multiply
                assert len(pres)==2
                if self.shape2val(node2shape[pres[0]][1][-1]) == self.shape2val(node2shape[pres[1]][1][-2]):
                    row = f"{out_var} = {in_var[0]} @ {in_var[1]}"
                elif self.shape2val(node2shape[pres[1]][1][-1]) == self.shape2val(node2shape[pres[0]][1][-2]):
                    row = f"{out_var} = {in_var[1]} @ {in_var[0]}"
                else:
                    raise DAGError(f"node {i} error: multiply operation's input can not multiply")
            elif op.lower() == 'droppath':
                # prob = float(params['prob'])
                # if prob>0:
                #     fun_forward = fun_forward + '\n' + tab*2+'if self.training:'
                #     fun_forward = fun_forward + '\n' + tab*3+f'{out_var}=drop_path({in_var},{prob})'
                # else:
                #     fun_forward = fun_forward + '\n' + tab*2+f'{out_var}={in_var}'
                fun_forward = fun_forward + '\n' + tab*2+'if self.training:'
                fun_forward = fun_forward + '\n' + tab*3+f'{out_var}=drop_path({in_var},drop_path_prob)'
                node2out_var[i] = out_var
                continue
            else:
                assert op=='output'
            node2out_var[i] = out_var
            fun_forward = fun_forward + '\n' + tab*2+row
        
        # All codes
        if self.mode=='nas-bench' or self.type=='stem' or self.mode=='detection':
            codes = f"""import torch
import torch.nn as nn
from torch.nn import (
    Conv2d,Linear,AvgPool2d,MaxPool2d,AdaptiveAvgPool2d,AdaptiveMaxPool2d,
    ReLU,GELU,Sigmoid,BatchNorm2d,LayerNorm
)
from {self.register_path} import Registers
from typing import *
import math

class CustomConv2d(nn.Module):

    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride:Union[int, Tuple[int, int]] = 1,
        dilation:Union[int, Tuple[int, int]] = 1,
        **kwargs):
        super().__init__()
        if isinstance(kernel_size,int):
            kernel_size=(kernel_size,kernel_size)
        if isinstance(stride,int):
            stride=(stride,stride)
        if isinstance(dilation,int):
            dilation=(dilation,dilation)
        self.padding_custom=((dilation[0]*(kernel_size[0]-1)+1-stride[0])/2,(dilation[1]*(kernel_size[1]-1)+1-stride[1])/2)
        self.padding_ceil = (max(math.ceil(self.padding_custom[0]),0),max(math.ceil(self.padding_custom[1]),0))
        self.padding_right = ((dilation[0]*(kernel_size[0]-1)+1)/2,(dilation[1]*(kernel_size[1]-1)+1)/2)
        self.conv2d = Conv2d(in_channels,out_channels,kernel_size,stride,self.padding_ceil,dilation,**kwargs)
    
    def forward(self, input):
        res = self.conv2d(input)
        if self.padding_ceil[0]==self.padding_right[0]:
            res =  res[:,:,:-1]
        if self.padding_ceil[1]==self.padding_right[1]:
            res =  res[:,:,:,:-1]
        return res

class CustomMaxPool2d(nn.Module):

    def __init__(self,
        kernel_size: Union[int, Tuple[int, int]],
        stride:Union[int, Tuple[int, int]] = 1,
        **kwargs):
        super().__init__()
        if isinstance(kernel_size,int):
            kernel_size=(kernel_size,kernel_size)
        if isinstance(stride,int):
            stride=(stride,stride)
        self.padding_custom=((kernel_size[0]-stride[0])/2,(kernel_size[1]-stride[1])/2)
        self.padding_ceil = (max(math.ceil(self.padding_custom[0]),0),max(math.ceil(self.padding_custom[1]),0))
        self.padding_right = (kernel_size[0]/2,kernel_size[1]/2)
        self.pool = MaxPool2d(kernel_size,stride,self.padding_ceil,**kwargs)

    def forward(self, input):
        res = self.pool(input)
        if self.padding_ceil[0]==self.padding_right[0]:
            res =  res[:,:,:-1]
        if self.padding_ceil[1]==self.padding_right[1]:
            res =  res[:,:,:,:-1]
        return res

class CustomAvgPool2d(nn.Module):

    def __init__(self,
        kernel_size: Union[int, Tuple[int, int]],
        stride:Union[int, Tuple[int, int]] = 1,
        **kwargs):
        super().__init__()
        if isinstance(kernel_size,int):
            kernel_size=(kernel_size,kernel_size)
        if isinstance(stride,int):
            stride=(stride,stride)
        self.padding_custom=((kernel_size[0]-stride[0])/2,(kernel_size[1]-stride[1])/2)
        self.padding_ceil = (max(math.ceil(self.padding_custom[0]),0),max(math.ceil(self.padding_custom[1]),0))
        self.padding_right = (kernel_size[0]/2,kernel_size[1]/2)
        self.pool = AvgPool2d(kernel_size,stride,self.padding_ceil,**kwargs)

    def forward(self, input):
        res = self.pool(input)
        if self.padding_ceil[0]==self.padding_right[0]:
            res =  res[:,:,:-1]
        if self.padding_ceil[1]==self.padding_right[1]:
            res =  res[:,:,:,:-1]
        return res

class CustomUpsample(nn.Module):

    def __init__(self, scale_factor: Union[int, float, Tuple[float, float]] = None, size: Union[int, Tuple[int, int]] = None, mode: str = 'nearest', **kwargs):
        super().__init__()
        if size is not None:
            self.module = nn.Upsample(size=size, mode=mode, **kwargs)
        else:
            if scale_factor is None:
                scale_factor = 2
            self.module = nn.Upsample(scale_factor=scale_factor, mode=mode, **kwargs)

    def forward(self, input):
        return self.module(input)

@Registers.block
class {block_name}(nn.Module):

    def __init__(self,in_channels,out_channels=None):
        super().__init__()
        self.out_channels = out_channels
{fun_init}

    def forward(self,x):
        B,in_channels,H,W = x.shape
        out_channels = self.out_channels
{fun_forward}
"""
        elif self.mode=='darts':
            codes = f"""import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import (
    Conv2d,Linear,AvgPool2d,MaxPool2d,AdaptiveAvgPool2d,AdaptiveMaxPool2d,
    ReLU,GELU,Sigmoid,BatchNorm2d,LayerNorm
)
from {self.register_path} import Registers
from typing import *
import math

class CustomConv2d(nn.Module):

    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride:Union[int, Tuple[int, int]] = 1,
        dilation:Union[int, Tuple[int, int]] = 1,
        **kwargs):
        super().__init__()
        if isinstance(kernel_size,int):
            kernel_size=(kernel_size,kernel_size)
        if isinstance(stride,int):
            stride=(stride,stride)
        if isinstance(dilation,int):
            dilation=(dilation,dilation)
        self.padding_custom=((dilation[0]*(kernel_size[0]-1)+1-stride[0])/2,(dilation[1]*(kernel_size[1]-1)+1-stride[1])/2)
        self.padding_ceil = (max(math.ceil(self.padding_custom[0]),0),max(math.ceil(self.padding_custom[1]),0))
        self.padding_right = ((dilation[0]*(kernel_size[0]-1)+1)/2,(dilation[1]*(kernel_size[1]-1)+1)/2)
        self.conv2d = Conv2d(in_channels,out_channels,kernel_size,stride,self.padding_ceil,dilation,**kwargs)
    
    def forward(self, input):
        res = self.conv2d(input)
        if self.padding_ceil[0]==self.padding_right[0]:
            res =  res[:,:,:-1]
        if self.padding_ceil[1]==self.padding_right[1]:
            res =  res[:,:,:,:-1]
        return res

class CustomMaxPool2d(nn.Module):

    def __init__(self,
        kernel_size: Union[int, Tuple[int, int]],
        stride:Union[int, Tuple[int, int]] = 1,
        **kwargs):
        super().__init__()
        if isinstance(kernel_size,int):
            kernel_size=(kernel_size,kernel_size)
        if isinstance(stride,int):
            stride=(stride,stride)
        self.padding_custom=((kernel_size[0]-stride[0])/2,(kernel_size[1]-stride[1])/2)
        self.padding_ceil = (max(math.ceil(self.padding_custom[0]),0),max(math.ceil(self.padding_custom[1]),0))
        self.padding_right = (kernel_size[0]/2,kernel_size[1]/2)
        self.pool = MaxPool2d(kernel_size,stride,self.padding_ceil,**kwargs)

    def forward(self, input):
        res = self.pool(input)
        if self.padding_ceil[0]==self.padding_right[0]:
            res =  res[:,:,:-1]
        if self.padding_ceil[1]==self.padding_right[1]:
            res =  res[:,:,:,:-1]
        return res

class CustomAvgPool2d(nn.Module):

    def __init__(self,
        kernel_size: Union[int, Tuple[int, int]],
        stride:Union[int, Tuple[int, int]] = 1,
        **kwargs):
        super().__init__()
        if isinstance(kernel_size,int):
            kernel_size=(kernel_size,kernel_size)
        if isinstance(stride,int):
            stride=(stride,stride)
        self.padding_custom=((kernel_size[0]-stride[0])/2,(kernel_size[1]-stride[1])/2)
        self.padding_ceil = (max(math.ceil(self.padding_custom[0]),0),max(math.ceil(self.padding_custom[1]),0))
        self.padding_right = (kernel_size[0]/2,kernel_size[1]/2)
        self.pool = AvgPool2d(kernel_size,stride,self.padding_ceil,**kwargs)

    def forward(self, input):
        res = self.pool(input)
        if self.padding_ceil[0]==self.padding_right[0]:
            res =  res[:,:,:-1]
        if self.padding_ceil[1]==self.padding_right[1]:
            res =  res[:,:,:,:-1]
        return res

class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        out = self.bn(out)
        return out

def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        # mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)).to(x.device)
        x.div_(keep_prob)
        x.mul_(mask)
    return x

@Registers.block
class {block_name}(nn.Module):

    def __init__(self,in_channels,out_channels,C_pre=None,reduction_pre=False):
        super().__init__()
        self.out_channels = out_channels
        if C_pre==None:
            C_pre = in_channels
        if reduction_pre:
            self.preprocess0 = FactorizedReduce(C_pre, out_channels)
        elif C_pre!=out_channels:
            self.preprocess0 = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_pre, out_channels, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.preprocess0 = nn.Identity()
        if in_channels!=out_channels:
            self.preprocess1 = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.preprocess1 = nn.Identity()
        in_channels = out_channels
{fun_init}
        
    def forward(self,x1,x2=None,drop_path_prob=0.0):
        if x2==None:
            x2=x1
        x1 = self.preprocess0(x1)
        x2 = self.preprocess1(x2)
        B,out_channels,H,W = x1.shape
{fun_forward}
"""
        with open(save_file_path,'w') as f:
            f.write(codes)
        import_module_from_path(block_name,save_file_path)

    def cal_params_flops(self,block_name,input_shape=(4,128,32,32)):
        # For detection blocks, inputs are often already in `dim` channels.
        # Use identical in/out channels to avoid artificial mismatches.
        if self.mode=='detection':
            model = Registers.block[block_name](input_shape[1],input_shape[1])
        else:
            model = Registers.block[block_name](input_shape[1],input_shape[1]*2)
        flops, params = profile(model,(torch.randn(input_shape),),verbose=False)
        # flops = '{:.2f}'.format(flops/(1000**3))
        # params = '{:.2f}'.format(params/(1000**2))
        return params,flops

    def check_dag(self,dag_nx):
        if not nx.is_directed_acyclic_graph(dag_nx):
            raise DAGError("The computation graph of the block is not a directed acyclic graph.")
        
        in_degrees = dag_nx.in_degree()
        out_degrees = dag_nx.out_degree()
        input_nodes = [node for node, degree in in_degrees if degree == 0]
        output_nodes = [node for node, degree in out_degrees if degree == 0]
        
        # ===== COMMON STRICT VALIDATION FOR ALL MODES (except darts) =====
        if self.mode == 'darts' and self.type in ['base','downsample']:
            # DARTS mode: special case with 2 inputs
            if len(input_nodes) != 2:
                raise DAGError('Must has two input nodes named input1 and input2.')
            if len(output_nodes) != 1 or dag_nx.nodes[output_nodes[0]]['value']!='output':
                raise DAGError('output must be the only output node.')
        else:
            # COMMON VALIDATION for both NAS-BENCH and DETECTION modes
            
            # 1. Check number of nodes (prevent overly complex graphs)
            num_nodes = len(dag_nx.nodes())
            if num_nodes < 3:
                raise DAGError(f'Block requires at least 3 nodes (input, operation, output), got {num_nodes}.')
            if num_nodes > 50:
                raise DAGError(f'Block allows maximum 50 nodes to prevent overly complex graphs, got {num_nodes}.')
            
            # 2. Check number of edges (prevent overly connected graphs)
            num_edges = len(dag_nx.edges())
            if num_edges > 100:
                raise DAGError(f'Block allows maximum 100 edges to prevent overly complex graphs, got {num_edges}.')
            
            # 3. Check input nodes (ONLY DIFFERENCE between detection and nas-bench)
            if self.mode == 'detection':
                # Detection: 1~4 inputs allowed
                if len(input_nodes) < 1 or len(input_nodes) > 4:
                    raise DAGError(f'Detection mode requires 1~4 input nodes (input_*), got {len(input_nodes)}.')
                # Validate input node names
                for n in input_nodes:
                    v = str(dag_nx.nodes[n]['value']).lower()
                    if not v.startswith('input'):
                        raise DAGError(f'All zero-indegree nodes must be inputs named like input_*, got: {v}')
            else:
                # NAS-Bench: exactly 1 input
                if len(input_nodes) != 1:
                    raise DAGError(f'NAS-Bench mode requires exactly 1 input node, got {len(input_nodes)}.')
                if dag_nx.nodes[input_nodes[0]]['value'] != 'input':
                    raise DAGError(f"Input node must be named 'input', got: {dag_nx.nodes[input_nodes[0]]['value']}")
            
            # 4. Check output nodes (must be exactly 1) - COMMON
            if len(output_nodes) != 1:
                raise DAGError(f'Block requires exactly 1 output node, got {len(output_nodes)}.')
            if dag_nx.nodes[output_nodes[0]]['value'].lower() != 'output':
                raise DAGError(f"Output node must be named 'output', got: {dag_nx.nodes[output_nodes[0]]['value']}")
            
            # 5. Check for isolated nodes (not connected to main graph) - COMMON
            for node in dag_nx.nodes():
                if node in input_nodes or node in output_nodes:
                    continue
                # Check if node is reachable from any input
                reachable_from_input = False
                for input_node in input_nodes:
                    if nx.has_path(dag_nx, input_node, node):
                        reachable_from_input = True
                        break
                if not reachable_from_input:
                    raise DAGError(f'Node {node} is isolated: not reachable from any input node.')
                
                # Check if node can reach output
                if not nx.has_path(dag_nx, node, output_nodes[0]):
                    raise DAGError(f'Node {node} is isolated: cannot reach output node.')
            
            # 6. Check node degree constraints - COMMON
            for node in dag_nx.nodes():
                if node in input_nodes:
                    continue
                if node in output_nodes:
                    in_deg = dag_nx.in_degree(node)
                    if in_deg == 0:
                        raise DAGError(f'Output node {node} has no incoming edges.')
                    continue
                
                # Check intermediate nodes
                in_deg = dag_nx.in_degree(node)
                out_deg = dag_nx.out_degree(node)
                
                if in_deg == 0 and out_deg == 0:
                    raise DAGError(f'Node {node} is completely isolated (0 in-degree, 0 out-degree).')
                
                # Prevent nodes with too many inputs (unlikely to be valid)
                if in_deg > 10:
                    raise DAGError(f'Node {node} has too many inputs ({in_deg}). Maximum 10 inputs allowed.')
                
                # Check operation validity
                node_val = dag_nx.nodes[node]['value']
                op_name = node_val.split('(')[0] if '(' in node_val else node_val
                
                # Validate operation is in allowed list or has valid pattern
                if op_name not in self.ops:
                    # Check if it's a valid operation pattern (e.g., "Conv2d(...)")
                    valid_pattern = False
                    for allowed_op in self.ops:
                        if op_name.startswith(allowed_op):
                            valid_pattern = True
                            break
                    if not valid_pattern:
                        raise DAGError(f'Node {node} uses unknown operation: {op_name}. Allowed operations: {", ".join(self.ops)}')
            
            # 7. Check graph complexity (max path length) - COMMON
            try:
                for input_node in input_nodes:
                    path_length = nx.shortest_path_length(dag_nx, input_node, output_nodes[0])
                    if path_length > 30:
                        raise DAGError(f'Graph is too deep: path from {input_node} to output has {path_length} nodes. Maximum 30 allowed.')
            except nx.NetworkXNoPath:
                raise DAGError(f'No path exists from input {input_node} to output.')
        
        # ===== END COMMON STRICT VALIDATION =====
        return True
    
    def check(self,txt,with_isomorphic=False):
        """
        res={'error':'...'}: error
        res=-1: not in database and no error
        res>=0: in database
        """
        # dag = self.parse_txt(txt)
        try:
            dag = self.parse_txt(txt,get_shape=False)
            dag_nx = self.dag2nx(dag)
            self.check_dag(dag_nx)
            dag['node2shape'],dag['node2output'] = self.get_shape(dag)
        except (DAGError,Exception) as e:
            if isinstance(e,DAGError):
                return {'error':str(e)}
            else:
                return {'error':'The block definition does not comply with the regulations.'}
        if with_isomorphic:
            id = self.check_isomorphic(dag_nx)
            return id
        return -1

    def add_block(self,txt,id):
        """
        should check first
        """
        dag = self.parse_txt(txt)
        # dag['node2shape'],dag['node2output'] = self.get_shape(dag)
        dag_nx = self.dag2nx(dag)
        block_name = f"{id}_{self.type}"
        code_path = os.path.join(self.code_dir,f"{block_name}.py")
        self.dag2code(dag,block_name=block_name,save_file_path=code_path)
        txt_path = os.path.join(self.txt_dir,f"{block_name}.txt")
        self.save_txt(txt,txt_path)
        self.drawDAG(dag,file_name=block_name,file_dir=self.image_dir)
        dag_nx_path = os.path.join(self.dag_nx_dir,f"{block_name}.gpickle")
        self.save_dag_nx(dag_nx,dag_nx_path)
        try:
            params,flops = self.cal_params_flops(block_name,input_shape=(4,128,32,32))
        except Exception as e:
            return {'error':str(e)}
        
        # ===== COMMON STRICT VALIDATION: Check params and flops limits (all modes except darts) =====
        if self.mode != 'darts':
            # Params limit: 100M parameters (100 million)
            params_millions = params / 1e6
            if params_millions > 100:
                return {'error':f'Block has too many parameters ({params_millions:.2f}M). Maximum 100M parameters allowed.'}
            
            # FLOPs limit: 1000G FLOPs (1 TFLOPs)
            flops_giga = flops / 1e9
            if flops_giga > 1000:
                return {'error':f'Block has too many FLOPs ({flops_giga:.2f}G). Maximum 1000G FLOPs allowed.'}
            
            # Minimum check: block should have reasonable complexity
            if params_millions < 0.001:
                return {'error':f'Block has too few parameters ({params_millions:.6f}M). Block seems trivial or incorrectly defined.'}
        
        dag['params'] = params
        dag['flops'] = flops
        dag_path = os.path.join(self.dag_dir,f"{block_name}.json")
        self.save_dag(dag,dag_path)
        self.drawDAG(dag,file_name=f"{block_name}_shape",file_dir=self.image_dir,draw_edge_label=True)
        if block_name in self.annos:
            print(f'Ovewrite {block_name}')
        self.annos[block_name] = {'name':dag['name'],'params':params,'flops':flops}
        self.save_anno()
        return block_name

    def delete_block(self,id):
        block_name = f"{id}_{self.type}"
        if block_name in self.annos:
            del self.annos[block_name]
        self.save_anno()


        
        






if __name__=='__main__':
    factory = BlockFactory('/home/SENSETIME/yangzekang/ModelGen/BlockFactoryV1/blocks')
    # s = factory.load_txt('/home/SENSETIME/yangzekang/ModelGen/BlockFactoryV1/blocks/txt/block_2.txt')
    # factory.add_block(s)
    s = """##ResNextBasicBlock_SE##\n0:input\n1:output\n2:Conv2d(out_channels=C/4,kernel_size=1,stride=1)\n3:BN\n4:ReLU\n5:Conv2d(out_channels=C/4,kernel_size=3,stride=1,groups=32)\n6:BN\n7:ReLU\n8:Conv2d(out_channels=C,kernel_size=1,stride=1)\n9:BN\n10:Add\n11:ReLU\n12:AdaptiveAvgPool2d(output_size=1)\n13:reshape(-1,C)\n14:Linear(out_channels=C/16)\n15:ReLU\n16:Linear(out_channels=C)\n17:Sigmoid\n18:reshape(-1,C,1,1)\n19:Mul\n0->2\n2->3\n3->4\n4->5\n5->6\n6->7\n7->8\n8->9\n9->10\n0->10\n10->11\n11->12\n12->13\n13->14\n14->15\n15->16\n16->17\n17->18\n11->19\n18->19\n19->1"""
    factory.check(s)