"""
自然语言指令解析模块
解析用户输入的自然语言导航指令，提取目标位置类型
"""

import re
from typing import Optional, Tuple


# 房间类型关键词映射
ROOM_KEYWORDS = {
    "客厅": ["客厅", "living room", "livingroom", "lounge", "起居室", "会客厅"],
    "卧室": ["卧室", "bedroom", "bed room", "睡房", "寝室", "卧房", "睡觉"],
    "厨房": ["厨房", "kitchen", "厨房间", "灶房"],
    "餐厅": ["餐厅", "dining room", "diningroom", "饭厅", "食堂", "餐室"],
    "浴室": ["浴室", "bathroom", "bath room", "卫生间", "洗手间", "厕所", "washroom", "toilet", "淋浴间"],
    "走廊": ["走廊", "hallway", "corridor", "过道", "走道", "通道"],
    "书房": ["书房", "study", "study room", "书斋", "工作室", "office"],
    "门厅": ["门厅", "entrance", "foyer", "hall", "玄关", "入口"],
    "阳台": ["阳台", "balcony", "露台"],
    "车库": ["车库", "garage", "车房"],
    "楼梯": ["楼梯", "stairs", "staircase", "stairway"],
    "地下室": ["地下室", "basement", "cellar"],
    "洗衣房": ["洗衣房", "laundry", "laundry room"],
    "储物间": ["储物间", "storage", "closet", "储藏室", "杂物间"],
}

# 动作意图关键词
ACTION_KEYWORDS = {
    "去": ["去", "走到", "前往", "到", "去往", "走向", "到达", "抵达", "find", "go to", "go", "walk to", "navigate to"],
    "找": ["找", "找到", "寻找", "搜索", "搜寻", "查找", "look for", "find", "search", "locate"],
    "探索": ["探索", "explore", "exploration", "看看", "参观", "巡视", "逛"],
}


def parse_instruction(instruction: str) -> Optional[dict]:
    """
    解析用户自然语言指令

    Args:
        instruction: 用户输入的自然语言指令

    Returns:
        解析结果字典，包含：
        - goal_room: 目标房间类型（中文）
        - goal_room_en: 目标房间类型（英文）
        - action_type: 动作类型（去/找/探索）
        - raw_text: 原始指令文本
        - confidence: 置信度 (0.0-1.0)
        如果无法解析，返回 None
    """
    if not instruction or not instruction.strip():
        return None

    instruction = instruction.strip().lower()

    # 检测动作类型
    action_type = "去"  # 默认动作
    for action, keywords in ACTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in instruction:
                action_type = action
                break
        if action_type != "去":
            break

    # 检测目标房间
    best_match = None
    best_confidence = 0.0

    for room_cn, keywords in ROOM_KEYWORDS.items():
        for keyword in keywords:
            if keyword in instruction:
                # 计算置信度：关键词长度越长，置信度越高
                confidence = len(keyword) / len(instruction) if instruction else 0.5
                confidence = min(confidence + 0.5, 1.0)  # 基础置信度 0.5

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = room_cn
                    break

    if best_match:
        return {
            "goal_room": best_match,
            "goal_room_en": _chinese_to_english(best_match),
            "action_type": action_type,
            "raw_text": instruction,
            "confidence": best_confidence,
        }

    # 如果没匹配到房间类型，尝试提取任何位置相关的词
    location_patterns = [
        (r"(?:去|到|走向|前往|走到)\s*(\S+)", 0.4),
        (r"(?:找|找到|寻找)\s*(\S+)", 0.3),
    ]

    for pattern, base_conf in location_patterns:
        match = re.search(pattern, instruction)
        if match:
            location = match.group(1)
            return {
                "goal_room": location,
                "goal_room_en": location,
                "action_type": action_type,
                "raw_text": instruction,
                "confidence": base_conf,
            }

    return None


def _chinese_to_english(room_cn: str) -> str:
    """中文房间类型转英文"""
    mapping = {
        "客厅": "living_room",
        "卧室": "bedroom",
        "厨房": "kitchen",
        "餐厅": "dining_room",
        "浴室": "bathroom",
        "走廊": "hallway",
        "书房": "study",
        "门厅": "entrance",
        "阳台": "balcony",
        "车库": "garage",
        "楼梯": "stairs",
        "地下室": "basement",
        "洗衣房": "laundry",
        "储物间": "storage",
    }
    return mapping.get(room_cn, room_cn)


def get_room_features(goal_room: str) -> list:
    """
    获取目标房间的特征物体列表

    Args:
        goal_room: 目标房间类型（中文）

    Returns:
        特征物体名称列表
    """
    room_features = {
        "客厅": ["sofa", "couch", "coffee_table", "TV", "television", "carpet", "armchair", "bookshelf"],
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
    return room_features.get(goal_room, [])


def get_help_text() -> str:
    """获取帮助文本"""
    return """
╔══════════════════════════════════════════════════════════════╗
║                    VLN 导航系统 - 使用帮助                    ║
╚══════════════════════════════════════════════════════════════╝

【支持的指令格式】
  直接说想去哪里，例如：
  • "去客厅"          → 导航到客厅
  • "找到厨房"        → 导航到厨房
  • "走到卧室"        → 导航到卧室
  • "找浴室"          → 导航到浴室
  • "探索走廊"        → 探索走廊区域

【支持的目标位置】
  客厅、卧室、厨房、餐厅、浴室、走廊、书房、
  门厅、阳台、车库、楼梯、地下室、洗衣房、储物间

【其他命令】
  • q / quit / exit   → 退出程序
  • help              → 显示此帮助信息

【提示】
  • 支持中英文混合输入
  • 支持自然语言表达，如"我想去客厅看看"
  • 如果语义标签可用，系统会自动定位目标位置
  • 如果语义标签不可用，系统会使用纯 AI 视觉导航
"""
