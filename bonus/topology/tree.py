from typing import List, Optional, Dict, Set
import logging
import math

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bonus/fault_tolerant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FT-TreeTopology")

class FaultTolerantTreeTopology:
    """支持容错的树形拓扑"""
    
    def __init__(self):
        self.name = "tree"
    
    def get_parent(self, rank: int, size: int, active_nodes: Set[int] = None) -> Optional[int]:
        """获取父节点，考虑节点故障"""
        if active_nodes is None:
            active_nodes = set(range(size))
        
        if rank == 0:
            return None  # 根节点没有父节点
        
        # 计算理论上的父节点
        parent = (rank - 1) // 2
        
        # 如果父节点失效，向上查找可用的祖先节点
        if parent not in active_nodes:
            # 递归查找可用祖先
            ancestor = parent
            while ancestor > 0 and ancestor not in active_nodes:
                ancestor = (ancestor - 1) // 2
            
            # 如果找到可用祖先，返回它；否则默认返回根节点（如果根节点活跃）
            if ancestor in active_nodes:
                return ancestor
            elif 0 in active_nodes:
                return 0
            else:
                # 所有祖先都失效，包括根节点，无法继续通信
                logger.error(f"No active parent or ancestor found for rank {rank}")
                return None
        
        return parent
    
    def get_children(self, rank: int, size: int, active_nodes: Set[int] = None) -> List[int]:
        """获取子节点，考虑节点故障"""
        if active_nodes is None:
            active_nodes = set(range(size))
        
        children = []
        left_child = 2 * rank + 1
        right_child = 2 * rank + 2
        
        if left_child < size and left_child in active_nodes:
            children.append(left_child)
        elif left_child < size:
            # 如果左子节点失效，尝试添加其子节点（如果存在且活跃）
            self._add_active_descendants(left_child, size, active_nodes, children)
        
        if right_child < size and right_child in active_nodes:
            children.append(right_child)
        elif right_child < size:
            # 如果右子节点失效，尝试添加其子节点（如果存在且活跃）
            self._add_active_descendants(right_child, size, active_nodes, children)
        
        return children
    
    def _add_active_descendants(self, node: int, size: int, active_nodes: Set[int], result: List[int]) -> None:
        """递归添加节点的活跃后代"""
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        if left_child < size:
            if left_child in active_nodes:
                result.append(left_child)
            else:
                self._add_active_descendants(left_child, size, active_nodes, result)
        
        if right_child < size:
            if right_child in active_nodes:
                result.append(right_child)
            else:
                self._add_active_descendants(right_child, size, active_nodes, result)
    
    def get_siblings(self, rank: int, size: int, active_nodes: Set[int] = None) -> List[int]:
        """获取兄弟节点，考虑节点故障"""
        if active_nodes is None:
            active_nodes = set(range(size))
        
        if rank == 0:
            return []  # 根节点没有兄弟
        
        parent = (rank - 1) // 2
        siblings = []
        
        # 检查左兄弟
        if rank % 2 == 0:  # 右子节点
            left_sibling = rank - 1
            if left_sibling in active_nodes:
                siblings.append(left_sibling)
        else:  # 左子节点
            right_sibling = rank + 1
            if right_sibling < size and right_sibling in active_nodes:
                siblings.append(right_sibling)
        
        return siblings
    
    def get_all_nodes(self, size: int, active_nodes: Set[int] = None) -> List[int]:
        """获取所有节点，考虑节点故障"""
        if active_nodes is None:
            active_nodes = set(range(size))
        
        return sorted(list(active_nodes))
    
    def get_path(self, source: int, dest: int, size: int, active_nodes: Set[int] = None) -> List[int]:
        """获取从源到目标的路径，考虑节点故障"""
        if active_nodes is None:
            active_nodes = set(range(size))
        
        if source not in active_nodes or dest not in active_nodes:
            return []  # 源或目标节点失效
        
        if source == dest:
            return [source]
        
        # 找到两个节点的最近公共祖先
        path = []
        path_to_source = self._path_to_root(source, size, active_nodes)
        path_to_dest = self._path_to_root(dest, size, active_nodes)
        
        # 找到最近公共祖先
        lca = None
        for node in path_to_source:
            if node in path_to_dest:
                lca = node
                break
        
        if lca is not None:
            # 构建从源到LCA的路径
            while path_to_source and path_to_source[0] != lca:
                path.append(path_to_source.pop(0))
            
            # 添加LCA
            path.append(lca)
            
            # 构建从LCA到目标的路径（反向）
            lca_index = path_to_dest.index(lca)
            reverse_path = path_to_dest[:lca_index]
            path.extend(reversed(reverse_path))
        
        return path
    
    def _path_to_root(self, node: int, size: int, active_nodes: Set[int]) -> List[int]:
        """计算从节点到根的路径，考虑节点故障"""
        path = [node]
        current = node
        
        while current != 0:
            parent = self.get_parent(current, size, active_nodes)
            if parent is None:
                break  # 没有可用的父节点
            
            path.append(parent)
            current = parent
        
        return path 