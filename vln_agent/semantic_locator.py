"""
语义定位模块
从 HM3D 的 .semantic.txt 解析房间位置，实现语义地图定位
"""

import logging
import math
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from dataset_selector import parse_semantic_txt, SemanticClass

logger = logging.getLogger(__name__)


# 房间类型 → 特征物体映射表
ROOM_FEATURES = {
    "客厅": ["sofa", "couch", "coffee_table", "tv", "television", "carpet", "armchair", "bookshelf"],
    "卧室": ["bed", "pillow", "wardrobe", "nightstand", "dresser", "mirror"],
    "厨房": ["cabinet", "fridge", "refrigerator", "oven", "stove", "sink", "microwave", "counter"],
    "餐厅": ["dining_table", "chair", "plate", "vase"],
    "浴室": ["toilet", "shower", "bathtub", "sink", "mirror"],
    "走廊": ["door", "corridor", "hallway"],
    "书房": ["desk", "bookshelf", "computer", "chair"],
    "门厅": ["door", "entrance", "hall"],
    "阳台": ["door", "window", "railing"],
    "车库": ["door", "car"],
    "楼梯": ["stairs", "staircase", "railing"],
    "地下室": ["stairs", "storage"],
    "洗衣房": ["washer", "dryer", "sink"],
    "储物间": ["shelf", "storage", "box"],
}


class SemanticLocator:
    """
    语义定位器
    利用 HM3D 语义标签定位目标房间位置
    """

    def __init__(self, semantic_txt_path: Optional[str] = None):
        """
        初始化语义定位器

        Args:
            semantic_txt_path: .semantic.txt 文件路径
        """
        self.semantic_txt_path = semantic_txt_path
        self.semantic_classes: List[SemanticClass] = []
        self.is_available = False

        # 尝试加载语义标签
        if semantic_txt_path:
            self._load_semantic()

    def _load_semantic(self):
        """加载语义标签文件"""
        try:
            self.semantic_classes = parse_semantic_txt(self.semantic_txt_path)
            self.is_available = len(self.semantic_classes) > 0
            if self.is_available:
                logger.info(f"成功加载语义标签: {len(self.semantic_classes)} 个类别")
            else:
                logger.warning("语义标签文件为空")
        except Exception as e:
            logger.warning(f"加载语义标签失败: {e}")
            self.is_available = False

    def find_room_features(self, goal_room: str) -> List[Dict[str, Any]]:
        """
        在语义标签中查找目标房间的特征物体

        Args:
            goal_room: 目标房间类型（中文）

        Returns:
            匹配到的特征物体列表，每项包含 name, id, color_hex, found
        """
        features = ROOM_FEATURES.get(goal_room, [])
        if not features:
            return []

        results = []
        for feature in features:
            found = False
            matched_id = None
            matched_color = None

            for sc in self.semantic_classes:
                # 模糊匹配：特征关键词出现在语义类别名称中
                if feature.lower() in sc.name.lower():
                    found = True
                    matched_id = sc.id
                    matched_color = sc.color_hex
                    break

            results.append({
                "name": feature,
                "id": matched_id,
                "color_hex": matched_color,
                "found": found,
            })

        return results

    def estimate_room_position(
        self,
        goal_room: str,
        pathfinder=None,
        agent_position: Optional[Tuple[float, float, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        估计目标房间的位置

        注意：HM3D 的 .semantic.txt 只提供类别名称和颜色映射，
        不直接提供空间位置信息。此方法通过特征物体推断房间位置。

        Args:
            goal_room: 目标房间类型
            pathfinder: Habitat pathfinder 实例（用于获取可导航区域）
            agent_position: 智能体当前位置

        Returns:
            位置信息字典，包含 center_x, center_z, width, depth 等
            如果无法定位，返回 None
        """
        if not self.is_available:
            return None

        features = self.find_room_features(goal_room)
        matched_features = [f for f in features if f["found"]]

        if not matched_features:
            logger.warning(f"未找到 {goal_room} 的特征物体匹配")
            return None

        # 如果有 pathfinder，获取可导航区域的中心作为目标位置
        if pathfinder is not None:
            try:
                # 获取可导航区域的边界
                nav_bounds = self._get_navigable_bounds(pathfinder)
                if nav_bounds:
                    center_x = (nav_bounds[0] + nav_bounds[2]) / 2
                    center_z = (nav_bounds[1] + nav_bounds[3]) / 2
                    width = nav_bounds[2] - nav_bounds[0]
                    depth = nav_bounds[3] - nav_bounds[1]

                    # 在可导航区域中采样一个目标点
                    goal_position = self._sample_goal_position(pathfinder, agent_position)

                    if goal_position:
                        # 计算距离
                        distance = 0.0
                        if agent_position:
                            dx = goal_position[0] - agent_position[0]
                            dz = goal_position[2] - agent_position[2]
                            distance = math.sqrt(dx * dx + dz * dz)

                        # 获取路径
                        path_points = []
                        num_waypoints = 0
                        if agent_position and pathfinder:
                            path = habitat_pathfinder_try_get_path(
                                pathfinder, agent_position, goal_position
                            )
                            if path:
                                path_points = path.points
                                num_waypoints = len(path.points)

                        estimated_steps = max(int(distance / 0.08), 1) if distance > 0 else 50

                        return {
                            "center_x": center_x,
                            "center_z": center_z,
                            "width": max(width, 2.0),
                            "depth": max(depth, 2.0),
                            "distance": distance,
                            "num_waypoints": num_waypoints,
                            "estimated_steps": estimated_steps,
                            "goal_position": goal_position,
                            "path_points": path_points,
                            "matched_features": matched_features,
                        }
            except Exception as e:
                logger.warning(f"获取导航区域信息失败: {e}")

        # 如果没有 pathfinder 或获取失败，返回基于特征数量的估计
        return {
            "center_x": 0.0,
            "center_z": 0.0,
            "width": 5.0,
            "depth": 5.0,
            "distance": 10.0,
            "num_waypoints": 0,
            "estimated_steps": 50,
            "goal_position": None,
            "path_points": [],
            "matched_features": matched_features,
        }

    def _get_navigable_bounds(self, pathfinder) -> Optional[Tuple[float, float, float, float]]:
        """
        获取可导航区域的边界

        Returns:
            (min_x, min_z, max_x, max_z) 或 None
        """
        try:
            bounds = pathfinder.get_bounds()
            return (bounds[0][0], bounds[0][2], bounds[1][0], bounds[1][2])
        except Exception:
            return None

    def _sample_goal_position(
        self,
        pathfinder,
        agent_position: Optional[Tuple[float, float, float]] = None,
        max_samples: int = 50,
    ) -> Optional[Tuple[float, float, float]]:
        """
        在可导航区域中采样一个目标位置

        Args:
            pathfinder: Habitat pathfinder
            agent_position: 智能体当前位置
            max_samples: 最大采样次数

        Returns:
            目标位置 (x, y, z) 或 None
        """
        try:
            bounds = pathfinder.get_bounds()
            min_x, min_y, min_z = bounds[0]
            max_x, max_y, max_z = bounds[1]

            for _ in range(max_samples):
                x = np.random.uniform(min_x, max_x)
                z = np.random.uniform(min_z, max_z)
                y = np.random.uniform(min_y, max_y)

                point = np.array([x, y, z])
                if pathfinder.is_navigable(point):
                    # 如果指定了当前位置，确保目标点有一定距离
                    if agent_position is not None:
                        dx = x - agent_position[0]
                        dz = z - agent_position[2]
                        dist = math.sqrt(dx * dx + dz * dz)
                        if dist < 2.0:  # 至少距离 2 米
                            continue
                    return (float(x), float(y), float(z))

            # 如果采样失败，返回边界中心
            center_x = (min_x + max_x) / 2
            center_z = (min_z + max_z) / 2
            center_y = (min_y + max_y) / 2
            return (center_x, center_y, center_z)

        except Exception as e:
            logger.warning(f"采样目标位置失败: {e}")
            return None

    def get_semantic_info(self, goal_room: str) -> Dict[str, Any]:
        """
        获取语义定位的完整信息（用于终端输出和日志）

        Args:
            goal_room: 目标房间类型

        Returns:
            语义定位信息字典
        """
        features = self.find_room_features(goal_room)
        matched = [f for f in features if f["found"]]

        return {
            "semantic_available": self.is_available,
            "num_categories": len(self.semantic_classes),
            "goal_room": goal_room,
            "matches": features,
            "num_matched": len(matched),
            "center_x": 0.0,
            "center_z": 0.0,
            "width": 0.0,
            "depth": 0.0,
            "distance": 0.0,
            "num_waypoints": 0,
            "estimated_steps": 0,
        }


def habitat_pathfinder_try_get_path(pathfinder, start, end):
    """
    尝试使用 Habitat pathfinder 获取路径
    这是一个安全包装，防止 pathfinder 不可用时崩溃
    """
    try:
        import habitat_sim
        start_np = habitat_sim.utils.common.to_habitat_position(start)
        end_np = habitat_sim.utils.common.to_habitat_position(end)
        path = habitat_sim.ShortestPath()
        path.requested_start = start_np
        path.requested_end = end_np
        if pathfinder.find_path(path):
            return path
        return None
    except Exception as e:
        logger.debug(f"路径规划失败: {e}")
        return None
