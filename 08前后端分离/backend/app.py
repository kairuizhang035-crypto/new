#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识图谱后端API服务
提供知识图谱数据的RESTful API接口
"""

import json
import os
from flask import Flask, jsonify, request, session, make_response
from werkzeug.security import check_password_hash, generate_password_hash
import time
import hmac
import hashlib
import base64
import socket
from flask_cors import CORS
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')
CORS(app)  # 允许跨域请求

# 数据文件路径 - 使用绝对路径
DATA_FILE_PATH = '/home/zkr/因果发现3/07分离/增强知识图谱完整结构.json'
ALLOWED_DIRS = ['/home/zkr/因果发现3/07分离']
UPLOAD_DIR = '/home/zkr/因果发现3/07分离/uploads'
RAW_DIR = '/home/zkr/因果发现3/07分离/原始数据'
SECURE_DIR = '/home/zkr/因果发现3/secure_credentials'
OLD_USERS_FILE = '/home/zkr/因果发现3/07分离/users.json'
OLD_USER_LOG_FILE = '/home/zkr/因果发现3/07分离/users.log'
USERS_FILE = '/home/zkr/因果发现3/secure_credentials/users.json'
USER_LOG_FILE = '/home/zkr/因果发现3/secure_credentials/users.log'
JWT_SECRET = os.environ.get('JWT_SECRET', 'dev-jwt-secret')
JWT_TTL_SECONDS = 7 * 24 * 3600

def ensure_dir(p):
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass

def is_allowed_path(p):
    try:
        ap = os.path.abspath(p)
        for d in ALLOWED_DIRS:
            if ap.startswith(os.path.abspath(d) + os.sep) or ap == os.path.abspath(d):
                return True
        return False
    except Exception:
        return False

def safe_filename(name):
    base = os.path.basename(name or '')
    base = base.replace('..', '').replace('/', '_').replace('\\', '_').strip()
    return base or 'upload'

def b64url(data_bytes):
    s = base64.urlsafe_b64encode(data_bytes).rstrip(b'=')
    return s.decode('utf-8')

def b64url_json(obj):
    return b64url(json.dumps(obj, separators=(',', ':')).encode('utf-8'))

def jwt_encode(payload, secret):
    header = {"alg": "HS256", "typ": "JWT"}
    segments = [b64url_json(header), b64url_json(payload)]
    signing_input = '.'.join(segments).encode('utf-8')
    signature = hmac.new(secret.encode('utf-8'), signing_input, hashlib.sha256).digest()
    return '.'.join(segments + [b64url(signature)])

def jwt_decode(token, secret):
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return None
        signing_input = (parts[0] + '.' + parts[1]).encode('utf-8')
        sig = base64.urlsafe_b64decode(parts[2] + '==')
        expected = hmac.new(secret.encode('utf-8'), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload_json = base64.urlsafe_b64decode(parts[1] + '==').decode('utf-8')
        payload = json.loads(payload_json)
        if 'exp' in payload and int(payload['exp']) < int(time.time()):
            return None
        return payload
    except Exception:
        return None

def get_auth_user():
    auth = request.headers.get('Authorization', '')
    token = None
    if auth.startswith('Bearer '):
        token = auth.split(' ', 1)[1]
    else:
        token = request.cookies.get('auth_token')
    payload = jwt_decode(token or '', JWT_SECRET)
    if not payload:
        return None
    username = payload.get('sub')
    users = read_users()
    if username in users:
        return username
    return None

def read_users():
    try:
        if not os.path.exists(USERS_FILE):
            return {}
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f) or {}
    except Exception:
        return {}

def write_users(data):
    try:
        os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
        try:
            os.chmod(os.path.dirname(USERS_FILE), 0o700)
        except Exception:
            pass
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data or {}, f, ensure_ascii=False, indent=2)
        try:
            os.chmod(USERS_FILE, 0o600)
        except Exception:
            pass
        return True
    except Exception as e:
        logger.error(f'写入用户文件失败: {e}')
        return False

def log_user_event(event, info):
    try:
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        os.makedirs(os.path.dirname(USER_LOG_FILE), exist_ok=True)
        try:
            os.chmod(os.path.dirname(USER_LOG_FILE), 0o700)
        except Exception:
            pass
        with open(USER_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'time': ts, 'event': event, 'info': info}, ensure_ascii=False) + '\n')
        try:
            os.chmod(USER_LOG_FILE, 0o600)
        except Exception:
            pass
    except Exception as e:
        logger.error(f'记录用户日志失败: {e}')

def get_server_ip():
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return '0.0.0.0'

def init_default_user():
    users = read_users()
    changed = False
    if 'zkr' not in users:
        users['zkr'] = {
            'password_hash': hashlib.sha256('123456'.encode('utf-8')).hexdigest(),
            'must_change_password': True,
            'created_at': int(time.time())
        }
        changed = True
    if 'root' not in users:
        users['root'] = {
            'password_hash': hashlib.sha256('root'.encode('utf-8')).hexdigest(),
            'must_change_password': False,
            'created_at': int(time.time())
        }
        changed = True
    if changed:
        write_users(users)
        ip = get_server_ip()
        logger.info(f'安全凭据初始化，路径: {USERS_FILE}')
        if 'zkr' in users:
            log_user_event('create_default_user', {'username': 'zkr', 'ip': ip})
        if 'root' in users:
            log_user_event('create_default_user', {'username': 'root', 'ip': ip})

class KnowledgeGraphAPI:
    def __init__(self, json_file_path=None):
        self.json_file_path = json_file_path or DATA_FILE_PATH
        self.data = None
        self.load_data()
        init_default_user()
    
    def load_data(self):
        """加载JSON数据"""
        try:
            # 确保使用绝对路径
            if not os.path.isabs(self.json_file_path):
                # 如果是相对路径，转换为绝对路径
                base_dir = os.path.dirname(os.path.abspath(__file__))
                self.json_file_path = os.path.join(base_dir, self.json_file_path)
            
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"成功加载数据文件: {self.json_file_path}")
            print(f"数据结构包含: {list(self.data.keys())}")
        except Exception as e:
            print(f"加载数据文件失败: {e}")
            self.data = {}
    
    def set_data_file(self, json_file_path):
        self.json_file_path = json_file_path or self.json_file_path
        self.load_data()
    
    def get_nodes(self):
        """获取节点数据 (V)"""
        return self.data.get('V', [])
    
    def get_edges(self):
        """获取边数据 (E_core)"""
        return self.data.get('E_core', [])
    
    def get_relations(self):
        """获取关系类型数据 (R)"""
        return self.data.get('R', {})
    
    def get_weights(self):
        """获取权重系统数据 (W)"""
        return self.data.get('W', {})

    def get_weight_details(self, key: str):
        """获取单个权重的详情。
        - 从 W 集合中读取指定键的权重对象；若不存在，返回 exists=False。
        - 统一返回结构，包含基础权重、候选/分层/三角权重与参数等字段（若存在）。
        - 附带该权重可能对应的边列表（通过 E_core 中的 weight_ref 反查）。
        """
        weights = self.get_weights() or {}
        wobj = None
        # 兼容字符串/数值键
        if key in weights:
            wobj = weights.get(key)
        elif str(key) in weights:
            wobj = weights.get(str(key))

        details = {
            'key': key,
            'exists': bool(wobj),
            'base_weight': {},
            'candidate_details': {},
            'hierarchy_weight': {},
            'triangulation_weights': {},
            'weight_params': {},
            'related_edges': []
        }

        if not wobj or not isinstance(wobj, dict):
            return details

        # 提取常见字段（与前端权重详情面板对应的命名）
        details['base_weight'] = wobj.get('base_weight') or {}
        details['candidate_details'] = wobj.get('candidate_details') or wobj.get('candidate_weight') or {}
        details['hierarchy_weight'] = wobj.get('hierarchy_weight') or {}
        details['triangulation_weights'] = wobj.get('triangulation_weights') or {}
        details['weight_params'] = wobj.get('weight_params') or {}

        # 反查相关边
        try:
            edges = self.get_edges() or []
            ref_key = key
            related = []
            for e in edges:
                if str(e.get('weight_ref')) == str(ref_key):
                    related.append({
                        'source': e.get('source'),
                        'target': e.get('target'),
                        'relation_type': e.get('relation_type'),
                        'edge_hierarchy': e.get('edge_hierarchy'),
                        'is_direct': bool(e.get('is_direct'))
                    })
            details['related_edges'] = related
        except Exception:
            pass

        return details
    
    def get_parameters(self):
        """获取参数学习数据 (Theta)"""
        return self.data.get('Theta', {})

    def search_parameters(self, query: str):
        """按关键字搜索参数(Θ)集合。优先在Theta键中搜索，必要时尝试方法文件。"""
        query = (query or '').strip().lower()
        params = self.get_parameters() or {}
        if not query:
            # 返回对象形式的摘要
            return { 'count': len(params), 'items': params }

        results = {}
        try:
            for k, v in (params.items() if isinstance(params, dict) else []):
                k_lower = str(k).lower()
                if query in k_lower:
                    results[k] = v
                    continue
                # 在方法名或字段上做一次浅匹配
                if isinstance(v, dict):
                    fields = ' '.join(list(v.keys()))
                    if query and query in fields.lower():
                        results[k] = v
        except Exception:
            pass

        # 若Theta缺失或命中很少，尝试从方法结果文件中补充（仅做键命中，不读取全部细节）
        if len(results) == 0:
            method_files = [
                '/home/zkr/因果发现3/03多方法参数学习/01MLE_CPT结果/MLE_CPTs.json',
                '/home/zkr/因果发现3/03多方法参数学习/02Bayesian_CPT结果/Bayesian_CPTs.json',
                '/home/zkr/因果发现3/03多方法参数学习/03EM_CPT结果/EM_CPTs.json',
                '/home/zkr/因果发现3/03多方法参数学习/04SEM_结果/SEM_CPT结果/SEM_CPTs.json'
            ]
            for fp in method_files:
                try:
                    if not os.path.exists(fp):
                        continue
                    with open(fp, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    # 兼容列表/字典两种结构
                    if isinstance(data, dict):
                        for k in data.keys():
                            if query in str(k).lower():
                                results.setdefault(k, {'from_file': os.path.basename(fp)})
                    elif isinstance(data, list):
                        # 若是列表，尝试每项中的name/id字段
                        for item in data:
                            key = item.get('name') or item.get('id') or item.get('variable')
                            if key and query in str(key).lower():
                                results.setdefault(key, {'from_file': os.path.basename(fp)})
                except Exception:
                    continue

        return { 'count': len(results), 'items': results }

    def get_parameter_details(self, param_id: str):
        """获取单个参数节点的详情。
        - 优先从 Theta 中读取（支持嵌套：Theta[param_id].parameter_learning.{MLE,Bayesian,EM,SEM}）。
        - 若不存在，尝试各方法结果文件做降级。
        - 统一返回结构，便于前端直接展示每个方法的 CPT、边级似然增益、稳定性等信息。
        """
        params = self.get_parameters() or {}
        param_obj = None
        if isinstance(params, dict):
            # 兼容字符串/数值键
            if param_id in params:
                param_obj = params.get(param_id)
            elif str(param_id) in params:
                param_obj = params.get(str(param_id))

        details = {
            'param_id': param_id,
            'exists': False,
            'source': 'Theta',
            # 方法可用性与摘要（兼容旧前端）
            'available_methods': {
                'MLE': False, 'Bayesian': False, 'EM': False, 'SEM': False
            },
            'method_summaries': {},
            # 扩展：前端直接渲染的结构化数据
            'methods': {},                 # 每方法原始/关键信息（含 cpt_data、parents、node_type 等）
            'cpt_info': {},                # 各方法 CPT 形状与是否存在
            'method_estimates': {},        # 各方法的似然增益/S_param/SEM估计等
            'parameter_stability': {},     # 稳定性总体与方法分数/相关性
            'edge_conditional_prob': {}    # 边条件依赖与父影响类型
        }

        def summarize_method(mname, mdata):
            summ = {'available': False}
            if mdata is None:
                return summ
            summ['available'] = True
            if isinstance(mdata, dict):
                # 提供键数量与前三个键名预览
                summ['entries'] = len(mdata)
                try:
                    keys = list(mdata.keys())
                    summ['preview_keys'] = keys[:3]
                except Exception:
                    pass
            elif isinstance(mdata, list):
                summ['rows'] = len(mdata)
                try:
                    summ['preview_rows'] = mdata[:2]
                except Exception:
                    pass
            else:
                # 其他标量，直接展示类型与值
                summ['type'] = type(mdata).__name__
                summ['value'] = mdata
            return summ

        if isinstance(param_obj, dict):
            # 兼容嵌套结构：Theta[param_id].parameter_learning.{MLE,Bayesian,EM,SEM}
            methods_obj = None
            if 'parameter_learning' in param_obj and isinstance(param_obj.get('parameter_learning'), dict):
                methods_obj = param_obj.get('parameter_learning')
                details['source'] = 'Theta.parameter_learning'
            else:
                methods_obj = param_obj

            details['exists'] = True
            details['available_methods'] = {
                'MLE': bool(methods_obj.get('MLE')),
                'Bayesian': bool(methods_obj.get('Bayesian')),
                'EM': bool(methods_obj.get('EM')),
                'SEM': bool(methods_obj.get('SEM')),
            }

            # 直接透传关键信息，前端可展示完整细节
            for m in ('MLE','Bayesian','EM','SEM'):
                mdata = methods_obj.get(m)
                if mdata is not None:
                    # 提取常见字段（兼容不同方法的数据结构）
                    method_payload = {}
                    if isinstance(mdata, dict):
                        # CPT、节点类型、父集合等
                        method_payload['cpt_data'] = mdata.get('cpt_data')
                        method_payload['node_type'] = mdata.get('node_type')
                        method_payload['parents'] = mdata.get('parents')
                        method_payload['has_complete_cpt'] = mdata.get('has_complete_cpt')
                        # SEM可能是系数等
                        for k in ('coefficient', 'intercept', 'r_squared', 'coefficient_std_error', 't_statistic', 'adjusted_r_squared', 'mse', 'rmse', 'data_quality'):
                            if k in mdata:
                                method_payload[k] = mdata.get(k)
                    else:
                        # 标量或列表，原样保存
                        method_payload['raw'] = mdata

                    details['methods'][m] = method_payload
                details['method_summaries'][m] = summarize_method(m, mdata)

            # 附加同层级信息（若存在）
            if isinstance(param_obj.get('cpt_info'), dict):
                details['cpt_info'] = param_obj.get('cpt_info')
            if isinstance(param_obj.get('method_estimates'), dict):
                details['method_estimates'] = param_obj.get('method_estimates')
            if isinstance(param_obj.get('parameter_stability'), dict):
                details['parameter_stability'] = param_obj.get('parameter_stability')
            if isinstance(param_obj.get('edge_conditional_prob'), dict):
                details['edge_conditional_prob'] = param_obj.get('edge_conditional_prob')

            # 附加：聚合边级似然增益，计算每方法与总体的平均值
            try:
                edge_gain_path = '/home/zkr/因果发现3/03多方法参数学习/05边级似然增益结果/边级似然增益结果.json'
                if os.path.exists(edge_gain_path):
                    with open(edge_gain_path, 'r', encoding='utf-8') as f:
                        edge_gain_data = json.load(f)

                    # 统一结构：method_estimates[m]['edge_likelihood_gain'] = { 'A->B': number }
                    # 并计算每方法平均与总体平均
                    overall_values = []
                    if isinstance(edge_gain_data, dict):
                        for mname, edges_obj in edge_gain_data.items():
                            if not isinstance(edges_obj, dict):
                                continue
                            gains_map = {}
                            method_values = []
                            for edge_key, rec in edges_obj.items():
                                # rec 形如 { full_likelihood, drop_likelihood, likelihood_gain, edge: [source, target], ... }
                                try:
                                    edge_pair = rec.get('edge') if isinstance(rec, dict) else None
                                    if isinstance(edge_pair, list) and len(edge_pair) == 2:
                                        src, tgt = edge_pair[0], edge_pair[1]
                                        if str(src) == str(param_id) or str(tgt) == str(param_id):
                                            gain = rec.get('likelihood_gain')
                                            if isinstance(gain, (int, float)):
                                                gains_map[edge_key] = gain
                                                method_values.append(gain)
                                                overall_values.append(gain)
                                    else:
                                        # 若无edge数组，则回退通过键匹配
                                        if (isinstance(edge_key, str) and (
                                            edge_key.startswith(f"{param_id}->") or edge_key.endswith(f"->{param_id}")
                                        )):
                                            gain = rec.get('likelihood_gain')
                                            if isinstance(gain, (int, float)):
                                                gains_map[edge_key] = gain
                                                method_values.append(gain)
                                                overall_values.append(gain)
                                except Exception:
                                    continue

                            if method_values:
                                details.setdefault('method_estimates', {})
                                details['method_estimates'].setdefault(mname, {})
                                details['method_estimates'][mname]['edge_likelihood_gain'] = gains_map
                                details['method_estimates'][mname]['likelihood_gain_avg'] = sum(method_values) / max(1, len(method_values))

                        # 同时提供按方法聚合的映射，便于前端另一种读取方式
                        if details.get('method_estimates'):
                            edge_lg_map = {}
                            for mname in list(details['method_estimates'].keys()):
                                mobj = details['method_estimates'].get(mname) or {}
                                if 'edge_likelihood_gain' in mobj:
                                    edge_lg_map[mname] = mobj['edge_likelihood_gain']
                            if edge_lg_map:
                                details['method_estimates']['edge_likelihood_gain'] = edge_lg_map

                            if overall_values:
                                details['method_estimates']['likelihood_gain_avg'] = sum(overall_values) / max(1, len(overall_values))
            except Exception:
                # 静默失败，不影响已有字段
                pass

            return details

        # Theta中未找到，回退到方法输出文件
        method_map = {
            'MLE': '/home/zkr/因果发现3/03多方法参数学习/01MLE_CPT结果/MLE_CPTs.json',
            'Bayesian': '/home/zkr/因果发现3/03多方法参数学习/02Bayesian_CPT结果/Bayesian_CPTs.json',
            'EM': '/home/zkr/因果发现3/03多方法参数学习/03EM_CPT结果/EM_CPTs.json',
            'SEM': '/home/zkr/因果发现3/03多方法参数学习/04SEM_结果/SEM_CPT结果/SEM_CPTs.json',
        }
        details['source'] = 'method_files'
        for mname, fp in method_map.items():
            try:
                if not os.path.exists(fp):
                    continue
                with open(fp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                found = None
                if isinstance(data, dict):
                    if param_id in data:
                        found = data.get(param_id)
                    elif str(param_id) in data:
                        found = data.get(str(param_id))
                elif isinstance(data, list):
                    for item in data:
                        key = item.get('name') or item.get('id') or item.get('variable')
                        if key == param_id or str(key) == str(param_id):
                            found = item
                            break
                if found is not None:
                    details['available_methods'][mname] = True
                    details['method_summaries'][mname] = summarize_method(mname, found)
                    # 统一映射为前端可直接渲染的结构
                    payload = {}
                    if isinstance(found, dict):
                        # 常见方法文件结构：{"type":"conditional","parents":[...],"probabilities":{...}}
                        # 前端支持的 cpt_data 形态：数组/对象/含 table 字段的对象/父状态到目标分布字典
                        cpt = (
                            found.get('cpt_data') or
                            found.get('cpt') or
                            found.get('probabilities') or
                            found
                        )
                        payload['cpt_data'] = cpt
                        # 透传父节点与类型，供状态栏展示与条件标签生成
                        parents = found.get('parents')
                        if isinstance(parents, list):
                            payload['parents'] = parents
                        ntype = found.get('node_type') or found.get('type')
                        if ntype is not None:
                            payload['node_type'] = ntype
                        # 简单完整性标记：存在概率或cpt字段视为完整
                        payload['has_complete_cpt'] = bool(found.get('probabilities') or found.get('cpt_data') or found.get('cpt'))
                    else:
                        # 标量/其他形态，保留原始值以便调试
                        payload['raw'] = found
                    details['methods'][mname] = payload
            except Exception:
                continue

        # Theta中未找到，已尝试方法文件。附加边级似然增益聚合
        try:
            edge_gain_path = '/home/zkr/因果发现3/03多方法参数学习/05边级似然增益结果/边级似然增益结果.json'
            if os.path.exists(edge_gain_path):
                with open(edge_gain_path, 'r', encoding='utf-8') as f:
                    edge_gain_data = json.load(f)

                overall_values = []
                if isinstance(edge_gain_data, dict):
                    for mname, edges_obj in edge_gain_data.items():
                        if not isinstance(edges_obj, dict):
                            continue
                        gains_map = {}
                        method_values = []
                        for edge_key, rec in edges_obj.items():
                            try:
                                edge_pair = rec.get('edge') if isinstance(rec, dict) else None
                                if isinstance(edge_pair, list) and len(edge_pair) == 2:
                                    src, tgt = edge_pair[0], edge_pair[1]
                                    if str(src) == str(param_id) or str(tgt) == str(param_id):
                                        gain = rec.get('likelihood_gain')
                                        if isinstance(gain, (int, float)):
                                            gains_map[edge_key] = gain
                                            method_values.append(gain)
                                            overall_values.append(gain)
                                else:
                                    if (isinstance(edge_key, str) and (
                                        edge_key.startswith(f"{param_id}->") or edge_key.endswith(f"->{param_id}")
                                    )):
                                        gain = rec.get('likelihood_gain')
                                        if isinstance(gain, (int, float)):
                                            gains_map[edge_key] = gain
                                            method_values.append(gain)
                                            overall_values.append(gain)
                            except Exception:
                                continue

                        if method_values:
                            details.setdefault('method_estimates', {})
                            details['method_estimates'].setdefault(mname, {})
                            details['method_estimates'][mname]['edge_likelihood_gain'] = gains_map
                            details['method_estimates'][mname]['likelihood_gain_avg'] = sum(method_values) / max(1, len(method_values))

                if details.get('method_estimates'):
                    edge_lg_map = {}
                    for mname in list(details['method_estimates'].keys()):
                        mobj = details['method_estimates'].get(mname) or {}
                        if 'edge_likelihood_gain' in mobj:
                            edge_lg_map[mname] = mobj['edge_likelihood_gain']
                    if edge_lg_map:
                        details['method_estimates']['edge_likelihood_gain'] = edge_lg_map
                    if overall_values:
                        details['method_estimates']['likelihood_gain_avg'] = sum(overall_values) / max(1, len(overall_values))
        except Exception:
            pass

        details['exists'] = any(details['available_methods'].values())
        return details
    
    def get_pathways(self):
        """获取路径分析数据 (Phi)"""
        return self.data.get('Phi', {})

    def get_pathways_for_list_view(self):
        """获取用于列表视图的路径分析摘要"""
        pathways = self.get_pathways() or {}
        summary_list = []

        for key, value in pathways.items():
            path_summary = {
                'key': key,
                'direct_effect': None,
                'indirect_effect': None,
                'total_effect': None,
                'confidence': None,
                'coverage': '—'
            }

            if isinstance(value, dict):
                if 'significance_info' in value and 'most_significant_pathway' in value['significance_info']:
                    sig_path_id_raw = value['significance_info']['most_significant_pathway']
                    path_summary['confidence'] = value['significance_info'].get('max_significance')

                    # 兼容数字/字符串键
                    med = value.get('mediation_effects', {}) or {}
                    sig_key = None
                    if sig_path_id_raw in med:
                        sig_key = sig_path_id_raw
                    elif str(sig_path_id_raw) in med:
                        sig_key = str(sig_path_id_raw)
                    elif isinstance(sig_path_id_raw, (int, float)) and int(sig_path_id_raw) in med:
                        sig_key = int(sig_path_id_raw)

                    if sig_key is not None:
                        sig_path_data = med[sig_key]
                        de = sig_path_data.get('direct_effect', {})
                        ie = sig_path_data.get('indirect_effect', {})
                        te = sig_path_data.get('total_effect', {})
                        path_summary['direct_effect'] = de.get('mean') if isinstance(de, dict) else None
                        path_summary['indirect_effect'] = ie.get('mean') if isinstance(ie, dict) else None
                        path_summary['total_effect'] = te.get('mean') if isinstance(te, dict) else None
            
            summary_list.append(path_summary)

        return summary_list

    def get_pathway_details(self, key):
        """获取单条路径的详情，统一格式输出"""
        pathways = self.get_pathways() or {}
        p = pathways.get(key)
        if p is None or not isinstance(p, dict):
            return {
                'key': key,
                'exists': False,
                'message': '路径不存在或数据格式不正确'
            }

        # Basic info from the most significant pathway（兼容数字/字符串ID）
        direct_effect, indirect_effect, total_effect, confidence = None, None, None, None
        most_sig_id_raw = None
        sig_path_obj = None
        med_effects = p.get('mediation_effects', {}) or {}
        if isinstance(p.get('significance_info'), dict):
            confidence = p['significance_info'].get('max_significance')
            most_sig_id_raw = p['significance_info'].get('most_significant_pathway')
            # 兼容字符串/数字键
            if most_sig_id_raw in med_effects:
                sig_path_obj = med_effects.get(most_sig_id_raw)
            elif str(most_sig_id_raw) in med_effects:
                sig_path_obj = med_effects.get(str(most_sig_id_raw))
            elif isinstance(most_sig_id_raw, (int, float)) and int(most_sig_id_raw) in med_effects:
                sig_path_obj = med_effects.get(int(most_sig_id_raw))
        if isinstance(sig_path_obj, dict):
            de = sig_path_obj.get('direct_effect', {}) or {}
            ie = sig_path_obj.get('indirect_effect', {}) or {}
            te = sig_path_obj.get('total_effect', {}) or {}
            direct_effect = de.get('mean')
            indirect_effect = ie.get('mean')
            total_effect = te.get('mean')

        # 组装核心/候选路径文本 & 详细中介路径列表
        core_paths = []
        candidate_paths = []
        mediation_effects_list = []
        if isinstance(med_effects, dict):
            for path_id, path_data in med_effects.items():
                # 路径描述转字符串
                desc = path_data.get('pathway_description')
                if isinstance(desc, list):
                    desc_str = ' → '.join([str(x) for x in desc])
                elif isinstance(desc, str):
                    desc_str = desc
                else:
                    desc_str = str(path_data.get('pathway_id') or path_id)

                # 核心/候选分类
                if path_data.get('is_significant'):
                    core_paths.append(desc_str)
                else:
                    candidate_paths.append(desc_str)

                # 详细记录（保留所有关键字段，前端可直接展示）
                record = {
                    'pathway_id': path_data.get('pathway_id', path_id),
                    'description': desc_str,
                    'is_significant': path_data.get('is_significant'),
                    'significance_probability': path_data.get('significance_probability'),
                    'mediation_ratio': path_data.get('mediation_ratio'),
                    'mediation_ratio_percentage': path_data.get('mediation_ratio_percentage'),
                    'primary_effect_type': path_data.get('primary_effect_type'),
                    'effect_strength': path_data.get('effect_strength'),
                    'indirect_effect_direction': path_data.get('indirect_effect_direction'),
                    'direct_effect_direction': path_data.get('direct_effect_direction'),
                    'mediation_type': path_data.get('mediation_type'),
                    'positive_effect_probability': path_data.get('positive_effect_probability'),
                    'negative_effect_probability': path_data.get('negative_effect_probability'),
                    'conclusion': path_data.get('conclusion'),
                    # 效应细节（完整对象，含mean/hdi/std）
                    'direct_effect': path_data.get('direct_effect', {}),
                    'indirect_effect': path_data.get('indirect_effect', {}),
                    'total_effect': path_data.get('total_effect', {}),
                }
                mediation_effects_list.append(record)

        # 显著优先、概率降序排序
        mediation_effects_list.sort(key=lambda r: (
            0 if r.get('is_significant') else 1,
            -(r.get('significance_probability') or 0)
        ))

        details = {
            'key': key,
            'exists': True,
            'path': key,
            'direct_effect': direct_effect,
            'indirect_effect': indirect_effect,
            'total_effect': total_effect,
            'confidence': confidence,
            'coverage': '—',
            'core_paths': core_paths,
            'candidate_paths': candidate_paths,
            # 扩展：原始与聚合信息，便于前端展示更多细节
            'significance_info': p.get('significance_info', {}),
            'pathway_membership': p.get('pathway_membership', []),
            'effect_statistics': p.get('effect_statistics', {}),
            'confidence_intervals': p.get('confidence_intervals', {}),
            'effect_directions': p.get('effect_directions', {}),
            'mediation_types': p.get('mediation_types', {}),
            'mediation_effects_list': mediation_effects_list,
            'most_significant_pathway_id': most_sig_id_raw,
        }

        return details

    def search_pathways(self, query):
        """按关键字搜索路径集合"""
        query = (query or '').strip().lower()
        pathways = self.get_pathways() or {}
        if not query:
            return { 'count': len(pathways), 'items': pathways }

        results = {}
        for k, v in pathways.items():
            k_lower = str(k).lower()
            hit = query in k_lower
            if not hit and isinstance(v, dict):
                for field in ('candidate_paths', 'core_paths'):
                    arr = v.get(field) or []
                    try:
                        if any(query in str(item).lower() for item in arr):
                            hit = True
                            break
                    except Exception:
                        pass
            elif not hit:
                try:
                    if any(query in str(item).lower() for item in (v if isinstance(v, list) else [v])):
                        hit = True
                except Exception:
                    pass

            if hit:
                results[k] = v

        return { 'count': len(results), 'items': results }
    
    def get_statistics(self):
        """获取统计信息"""
        nodes = self.get_nodes()
        edges = self.get_edges()
        relations = self.get_relations()
        weights = self.get_weights()
        parameters = self.get_parameters()
        pathways = self.get_pathways()
        
        # 统计节点类型
        node_types = {}
        for node in nodes:
            node_type = node.split('_')[0] if '_' in node else 'other'
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # 统计边类型
        edge_types = {}
        edge_hierarchies = {}
        for edge in edges:
            relation_type = edge.get('relation_type', 'unknown')
            edge_types[relation_type] = edge_types.get(relation_type, 0) + 1
            
            hierarchy = edge.get('edge_hierarchy', 'unknown')
            edge_hierarchies[hierarchy] = edge_hierarchies.get(hierarchy, 0) + 1
        
        return {
            'nodes': len(nodes),
            'edges': len(edges),
            'relations': len(relations),
            'weights': len(weights),
            'parameters': len(parameters),
            'pathways': len(pathways),
            'node_types': node_types,
            'edge_types': edge_types,
            'edge_hierarchies': edge_hierarchies
        }
    
    def get_node_types_detailed(self):
        """获取详细的节点类型统计"""
        nodes = self.get_nodes()
        edges = self.get_edges()
        
        node_stats = {}
        
        for node in nodes:
            node_type = node.split('_')[0] if '_' in node else 'other'
            if node_type not in node_stats:
                node_stats[node_type] = {
                    'count': 0,
                    'nodes': [],
                    'in_degree': 0,
                    'out_degree': 0,
                    'total_degree': 0
                }
            
            node_stats[node_type]['count'] += 1
            
            # 计算该节点的度数
            in_degree = sum(1 for edge in edges if edge['target'] == node)
            out_degree = sum(1 for edge in edges if edge['source'] == node)
            total_degree = in_degree + out_degree

            # 生成节点详情对象
            node_label = node.split('_')[1] if '_' in node else node
            node_info = {
                'id': node,
                'name': node_label,
                'type': node_type,
                'in_degree': in_degree,
                'out_degree': out_degree,
                'total_degree': total_degree
            }

            node_stats[node_type]['nodes'].append(node_info)
            
            # 汇总该类型的度数
            node_stats[node_type]['in_degree'] += in_degree
            node_stats[node_type]['out_degree'] += out_degree
            node_stats[node_type]['total_degree'] += total_degree
        
        # 计算平均度数
        for node_type in node_stats:
            count = node_stats[node_type]['count']
            if count > 0:
                node_stats[node_type]['avg_in_degree'] = node_stats[node_type]['in_degree'] / count
                node_stats[node_type]['avg_out_degree'] = node_stats[node_type]['out_degree'] / count
                node_stats[node_type]['avg_total_degree'] = node_stats[node_type]['total_degree'] / count
            
            # 将该类型下的节点按总度数排序，便于前端展示
            node_stats[node_type]['nodes'].sort(key=lambda n: n.get('total_degree', 0), reverse=True)
        
        return node_stats
    
    def search_nodes(self, query):
        """搜索节点"""
        nodes = self.get_nodes()
        if not query:
            return nodes
        
        query = query.lower()
        return [node for node in nodes if query in node.lower()]
    
    def get_node_details(self, node_id):
        """获取节点详细信息"""
        edges = self.get_edges()
        weights = self.get_weights()
        
        # 找到与该节点相关的边
        related_edges = []
        for edge in edges:
            if edge['source'] == node_id or edge['target'] == node_id:
                edge_key = edge.get('weight_ref', f"{edge['source']}->{edge['target']}")
                edge_weight = weights.get(edge_key, {})
                edge_info = {
                    **edge,
                    'weight_info': edge_weight
                }
                related_edges.append(edge_info)
        
        return {
            'node_id': node_id,
            'node_type': node_id.split('_')[0] if '_' in node_id else 'unknown',
            'related_edges': related_edges,
            'edge_count': len(related_edges)
        }

    def get_edge_details(self, source, target):
        """获取边详细信息：基础边、权重、关系元数据与候选同源边"""         
        edges = self.get_edges()
        weights = self.get_weights()
        relations = self.get_relations()

        # 找到与该源->目标匹配的所有边（可能存在多条不同关系/层次）
        matches = [e for e in edges if e.get('source') == source and e.get('target') == target]
        if not matches:
            return {
                'base': None,
                'weight': {},
                'relation': {},
                'alternatives': []
            }

        base = matches[0]
        weight_ref = base.get('weight_ref')
        weight = weights.get(weight_ref, {}) if weight_ref else {}
        relation_meta = relations.get(base.get('relation_type', ''), {})

        return {
            'base': base,
            'weight': weight,
            'relation': relation_meta,
            'alternatives': matches[1:]
        }

    def get_relation_stats_detailed(self):
        """按关系类型统计详细信息：总边数、各层次计数、直/间接计数及示例边"""
        edges = self.get_edges()
        stats = {}

        for e in edges:
            rt = e.get('relation_type', 'unknown')
            h = e.get('edge_hierarchy', 'unknown')
            is_direct = bool(e.get('is_direct', False))

            if rt not in stats:
                stats[rt] = {
                    'total': 0,
                    'triangulated_verified': 0,
                    'non_triangulated': 0,
                    'candidate_only': 0,
                    'unknown_hierarchy': 0,
                    'direct': 0,
                    'indirect': 0,
                    'examples': []
                }

            s = stats[rt]
            s['total'] += 1

            if h in ('triangulated_verified', 'non_triangulated', 'candidate_only'):
                s[h] += 1
            else:
                s['unknown_hierarchy'] += 1

            if is_direct:
                s['direct'] += 1
            else:
                s['indirect'] += 1

            # 收集所有示例边（完整列表，前端分页显示）
            s['examples'].append({
                'source': e.get('source'),
                'target': e.get('target'),
                'edge_hierarchy': h,
                'is_direct': is_direct
            })

        return stats

# 初始化API实例
kg_api = KnowledgeGraphAPI(DATA_FILE_PATH)

@app.route('/', methods=['GET'])
def index():
    """根路径欢迎页面"""
    return jsonify({
        'message': '知识图谱可视化后端API服务',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'health': '/api/health',
            'nodes': '/api/nodes',
            'node_types': '/api/nodes/types',
            'nodes_by_type': '/api/nodes/types/<node_type>',
            'edges': '/api/edges', 
            'edge_details': '/api/edges/<source>/<target>/details',
            'relations': '/api/relations',
            'weights': '/api/weights',
            'weight_details': '/api/weights/<key>/details',
            'parameters': '/api/parameters',
            'pathways': '/api/pathways',
            'statistics': '/api/statistics',
            'search_nodes': '/api/search/nodes?q=<query>',
            'node_details': '/api/nodes/<node_id>/details',
            'graph_data': '/api/graph/data'
        },
        'frontend_url': 'http://localhost:8080'
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'message': '知识图谱API服务运行正常'
    })

@app.route('/api/datasource/current', methods=['GET'])
def datasource_current():
    try:
        info = {
            'path': kg_api.json_file_path,
            'keys': list((kg_api.data or {}).keys())
        }
        return jsonify({'success': True, 'data': info})
    except Exception as e:
        logger.error(f"获取当前数据源失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/datasource/list', methods=['GET'])
def datasource_list():
    try:
        user = get_auth_user()
        if not user:
            return jsonify({'success': False, 'error': '未认证'}), 401
        ensure_dir(UPLOAD_DIR)
        seen = set()
        files = []
        for base in ALLOWED_DIRS:
            if not os.path.exists(base):
                continue
            for root, _, fnames in os.walk(base):
                for fn in fnames:
                    if fn.lower().endswith('.json'):
                        fp = os.path.join(root, fn)
                        ap = os.path.abspath(fp)
                        if ap in seen:
                            continue
                        seen.add(ap)
                        try:
                            st = os.stat(ap)
                            files.append({'path': ap, 'name': fn, 'size': st.st_size})
                        except Exception:
                            files.append({'path': ap, 'name': fn, 'size': None})
        return jsonify({'success': True, 'data': files})
    except Exception as e:
        logger.error(f"列出数据源失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/datasource/select', methods=['POST'])
def datasource_select():
    try:
        user = get_auth_user()
        if not user:
            return jsonify({'success': False, 'error': '未认证'}), 401
        payload = request.get_json(silent=True) or {}
        path = payload.get('path')
        if not path or not os.path.exists(path):
            return jsonify({'success': False, 'error': '文件不存在'}), 400
        if not is_allowed_path(path):
            return jsonify({'success': False, 'error': '不允许的路径'}), 400
        if not path.lower().endswith('.json'):
            return jsonify({'success': False, 'error': '仅支持JSON'}), 400
        kg_api.set_data_file(path)
        stats = kg_api.get_statistics()
        return jsonify({'success': True, 'data': {'path': kg_api.json_file_path, 'statistics': stats}})
    except Exception as e:
        logger.error(f"选择数据源失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/datasource/upload', methods=['POST'])
def datasource_upload():
    try:
        user = get_auth_user()
        if not user:
            return jsonify({'success': False, 'error': '未认证'}), 401
        f = request.files.get('file')
        if not f:
            return jsonify({'success': False, 'error': '缺少文件'}), 400
        fname = safe_filename(f.filename)
        ext = os.path.splitext(fname)[1].lower()
        if ext not in ('.json', '.csv'):
            return jsonify({'success': False, 'error': '仅支持CSV或JSON'}), 400
        target_dir = UPLOAD_DIR if ext == '.json' else RAW_DIR
        ensure_dir(target_dir)
        dest = os.path.join(target_dir, fname)
        if os.path.exists(dest):
            name, ext2 = os.path.splitext(fname)
            dest = os.path.join(target_dir, f"{name}_{int(__import__('time').time())}{ext2}")
        try:
            f.save(dest)
        except Exception as e:
            logger.error(f"保存文件失败: {e}")
            return jsonify({'success': False, 'error': '存储失败'}), 500
        auto_select = request.form.get('select', 'false').lower() in ('1','true','yes','on')
        selected = None
        if auto_select and ext == '.json':
            if is_allowed_path(dest):
                kg_api.set_data_file(dest)
                selected = {'path': kg_api.json_file_path}
        return jsonify({'success': True, 'data': {'saved_path': dest, 'selected': selected}})
    except Exception as e:
        logger.error(f"上传数据源失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/datasource/delete', methods=['POST'])
def datasource_delete():
    try:
        user = get_auth_user()
        if not user:
            return jsonify({'success': False, 'error': '未认证'}), 401
        payload = request.get_json(silent=True) or {}
        path = payload.get('path')
        if not path:
            return jsonify({'success': False, 'error': '缺少路径'}), 400
        ap = os.path.abspath(path)
        uploads_root = os.path.abspath(UPLOAD_DIR)
        if not ap.startswith(uploads_root + os.sep):
            return jsonify({'success': False, 'error': '仅允许删除上传目录文件'}), 400
        if not os.path.exists(ap):
            return jsonify({'success': False, 'error': '文件不存在'}), 404
        try:
            os.remove(ap)
        except Exception as e:
            return jsonify({'success': False, 'error': f'删除失败: {e}'}), 500
        current = kg_api.json_file_path
        if os.path.abspath(current) == ap:
            kg_api.set_data_file(DATA_FILE_PATH)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"删除数据源失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/nodes', methods=['GET'])
def get_nodes():
    """获取所有节点"""
    try:
        nodes = kg_api.get_nodes()
        return jsonify({
            'success': True,
            'data': nodes,
            'count': len(nodes)
        })
    except Exception as e:
        logger.error(f"获取节点失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/edges', methods=['GET'])
def get_edges():
    """获取所有边"""
    try:
        edges = kg_api.get_edges()
        return jsonify({
            'success': True,
            'data': edges,
            'count': len(edges)
        })
    except Exception as e:
        logger.error(f"获取边失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/edges/<source>/<target>/details', methods=['GET'])
def get_edge_details_route(source, target):
    """获取边详细信息"""
    try:
        details = kg_api.get_edge_details(source, target)
        return jsonify({
            'success': True,
            'data': details
        })
    except Exception as e:
        logger.error(f"获取边详细信息失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/relations', methods=['GET'])
def get_relations():
    """获取关系类型"""
    try:
        relations = kg_api.get_relations()
        return jsonify({
            'success': True,
            'data': relations
        })
    except Exception as e:
        logger.error(f"获取关系类型失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/relations/stats', methods=['GET'])
def get_relation_stats():
    """获取关系类型的详细统计"""
    try:
        stats = kg_api.get_relation_stats_detailed()
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        logger.error(f"获取关系类型统计失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/weights', methods=['GET'])
def get_weights():
    """获取权重系统"""
    try:
        weights = kg_api.get_weights()
        return jsonify({
            'success': True,
            'data': weights
        })
    except Exception as e:
        logger.error(f"获取权重系统失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/weights/<path:key>/details', methods=['GET'])
def get_weight_details_route(key):
    """获取单个权重详情"""
    try:
        details = kg_api.get_weight_details(key)
        return jsonify({
            'success': True,
            'data': details
        })
    except Exception as e:
        logger.error(f"获取权重详情失败: {e}")
        return jsonify({
            'success': False,
             'error': str(e)
        }), 500

@app.route('/api/parameters', methods=['GET'])
def get_parameters():
    """获取参数学习数据"""
    try:
        parameters = kg_api.get_parameters()
        return jsonify({
            'success': True,
            'data': parameters
        })
    except Exception as e:
        logger.error(f"获取参数学习数据失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/parameters/<path:param_id>/details', methods=['GET'])
def get_parameter_details_route(param_id):
    """获取单个参数节点的详情"""
    try:
        details = kg_api.get_parameter_details(param_id)
        return jsonify({
            'success': True,
            'data': details
        })
    except Exception as e:
        logger.error(f"获取参数详情失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/search/parameters', methods=['GET'])
def search_parameters_route():
    """搜索参数集合(Θ)"""
    try:
        query = request.args.get('q', '')
        result = kg_api.search_parameters(query)
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        logger.error(f"搜索参数失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/pathways', methods=['GET'])
def get_pathways():
    """获取路径分析数据"""
    try:
        pathways = kg_api.get_pathways_for_list_view()
        return jsonify({
            'success': True,
            'data': pathways
        })
    except Exception as e:
        logger.error(f"获取路径分析数据失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/pathways/<path:key>/details', methods=['GET'])
def get_pathway_details_route(key):
    """获取单条路径详情"""
    try:
        details = kg_api.get_pathway_details(key)
        return jsonify({
            'success': True,
            'data': details
        })
    except Exception as e:
        logger.error(f"获取路径详情失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/search/pathways', methods=['GET'])
def search_pathways_route():
    """搜索路径集合"""
    try:
        query = request.args.get('q', '')
        result = kg_api.search_pathways(query)
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        logger.error(f"搜索路径失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """获取统计信息"""
    try:
        stats = kg_api.get_statistics()
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/search/nodes', methods=['GET'])
def search_nodes():
    """搜索节点"""
    try:
        query = request.args.get('q', '')
        nodes = kg_api.search_nodes(query)
        return jsonify({
            'success': True,
            'data': nodes,
            'count': len(nodes),
            'query': query
        })
    except Exception as e:
        logger.error(f"搜索节点失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/nodes/<node_id>/details', methods=['GET'])
def get_node_details(node_id):
    """获取节点详细信息"""
    try:
        details = kg_api.get_node_details(node_id)
        return jsonify({
            'success': True,
            'data': details
        })
    except Exception as e:
        logger.error(f"获取节点详细信息失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/nodes/types', methods=['GET'])
def get_node_types():
    """获取节点类型详细统计"""
    try:
        node_types = kg_api.get_node_types_detailed()
        return jsonify({
            'success': True,
            'data': node_types
        })
    except Exception as e:
        logger.error(f"获取节点类型统计失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/nodes/types/<node_type>', methods=['GET'])
def get_nodes_by_type(node_type):
    """根据类型获取节点列表"""
    try:
        nodes = kg_api.get_nodes()
        filtered_nodes = [node for node in nodes if node.startswith(f"{node_type}_")]
        
        # 获取每个节点的基本统计信息
        edges = kg_api.get_edges()
        node_details = []
        
        for node in filtered_nodes:
            in_degree = sum(1 for edge in edges if edge['target'] == node)
            out_degree = sum(1 for edge in edges if edge['source'] == node)
            
            node_details.append({
                'id': node,
                'name': node,
                'type': node_type,
                'in_degree': in_degree,
                'out_degree': out_degree,
                'total_degree': in_degree + out_degree
            })
        
        # 按总度数排序
        node_details.sort(key=lambda x: x['total_degree'], reverse=True)
        
        return jsonify({
            'success': True,
            'data': node_details,
            'count': len(node_details),
            'type': node_type
        })
    except Exception as e:
        logger.error(f"根据类型获取节点失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/graph/data', methods=['GET'])
def get_graph_data():
    """获取完整的图数据用于可视化"""
    try:
        user = get_auth_user()
        if not user:
            return jsonify({'success': False, 'error': '未认证'}), 401
        nodes = kg_api.get_nodes()
        edges = kg_api.get_edges()
        relations = kg_api.get_relations()
        
        # 转换为vis-network格式
        vis_nodes = []
        for i, node in enumerate(nodes):
            node_type = node.split('_')[0] if '_' in node else 'unknown'
            vis_nodes.append({
                'id': node,
                'label': node.split('_')[1] if '_' in node else node,
                'group': node_type,
                'title': node
            })
        
        vis_edges = []
        for edge in edges:
            vis_edges.append({
                'from': edge['source'],
                'to': edge['target'],
                'label': relations.get(edge.get('relation_type', ''), {}).get('name', ''),
                'title': f"{edge['source']} -> {edge['target']}",
                'color': {'color': '#848484'},
                'arrows': 'to'
            })
        
        return jsonify({
            'success': True,
            'data': {
                'nodes': vis_nodes,
                'edges': vis_edges
            }
        })
    except Exception as e:
        logger.error(f"获取图数据失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# 简单认证机制（示例）：用户名密码保存在内存，实际应接入数据库/LDAP/OAuth
USERS = {
    'admin': generate_password_hash('admin123')
}

@app.route('/api/auth/login', methods=['POST'])
def login_route():
    try:
        payload = request.get_json(silent=True) or {}
        username = (payload.get('username') or '').strip()
        password = (payload.get('password') or '').strip()
        if not username or not password:
            return jsonify({'success': False, 'error': '用户名或密码不能为空'}), 400
        users = read_users()
        user = users.get(username)
        if not user:
            return jsonify({'success': False, 'error': '用户名或密码错误'}), 401
        pw_ok = hashlib.sha256(password.encode('utf-8')).hexdigest() == user.get('password_hash')
        if not pw_ok:
            return jsonify({'success': False, 'error': '用户名或密码错误'}), 401
        session['user'] = username
        now = int(time.time())
        token = jwt_encode({'sub': username, 'iat': now, 'exp': now + JWT_TTL_SECONDS}, JWT_SECRET)
        resp = make_response(jsonify({'success': True, 'user': {'name': username}, 'must_change_password': bool(user.get('must_change_password'))}))
        resp.set_cookie('auth_token', token, httponly=True, samesite='Lax', secure=False, path='/')
        return resp
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout_route():
    try:
        session.pop('user', None)
        resp = make_response(jsonify({'success': True}))
        resp.set_cookie('auth_token', '', httponly=True, samesite='Lax', secure=False, path='/', expires=0)
        return resp
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/auth/me', methods=['GET'])
def me_route():
    token = request.cookies.get('auth_token')
    payload = jwt_decode(token or '', JWT_SECRET)
    if not payload:
        return jsonify({'success': False, 'authenticated': False}), 200
    username = payload.get('sub')
    users = read_users()
    if username not in users:
        return jsonify({'success': False, 'authenticated': False}), 200
    return jsonify({'success': True, 'authenticated': True, 'user': {'name': username}})

@app.route('/api/auth/change_password', methods=['POST'])
def change_password_route():
    token = request.cookies.get('auth_token')
    payload = jwt_decode(token or '', JWT_SECRET)
    if not payload:
        return jsonify({'success': False, 'error': '未认证'}), 401
    username = payload.get('sub')
    body = request.get_json(silent=True) or {}
    old_pw = (body.get('old_password') or '').strip()
    new_pw = (body.get('new_password') or '').strip()
    if not old_pw or not new_pw:
        return jsonify({'success': False, 'error': '请输入原密码与新密码'}), 400
    users = read_users()
    user = users.get(username)
    if not user:
        return jsonify({'success': False, 'error': '用户不存在'}), 400
    if hashlib.sha256(old_pw.encode('utf-8')).hexdigest() != user.get('password_hash'):
        return jsonify({'success': False, 'error': '原密码错误'}), 400
    if len(new_pw) < 6:
        return jsonify({'success': False, 'error': '新密码长度至少6位'}), 400
    user['password_hash'] = hashlib.sha256(new_pw.encode('utf-8')).hexdigest()
    user['must_change_password'] = False
    users[username] = user
    ok = write_users(users)
    if not ok:
        return jsonify({'success': False, 'error': '保存失败'}), 500
    log_user_event('change_password', {'username': username, 'ip': request.remote_addr})
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
