"""
终端输出格式化模块
提供带 ANSI 颜色的格式化终端输出
"""

import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List


# ANSI 颜色码
class Colors:
    """ANSI 颜色常量"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # 前景色
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"

    # 背景色
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    # 高亮
    HIGHLIGHT_RED = "\033[41m\033[97m"
    HIGHLIGHT_GREEN = "\033[42m\033[97m"
    HIGHLIGHT_YELLOW = "\033[43m\033[30m"


class TerminalPrinter:
    """
    终端输出格式化器
    提供带颜色和格式的终端输出方法
    """

    def __init__(self, use_color: bool = True):
        """
        初始化终端输出器

        Args:
            use_color: 是否使用 ANSI 颜色
        """
        self.use_color = use_color
        self._start_time: Optional[datetime] = None

    def _c(self, color: str, text: str) -> str:
        """应用颜色（如果启用）"""
        if self.use_color:
            return f"{color}{text}{Colors.RESET}"
        return text

    def print_header(self, scene_name: str, goal: str, mode: str):
        """打印系统标题头"""
        line = "═" * 70
        self._print_line()
        print(self._c(Colors.BOLD + Colors.WHITE, f"  Habitat-VLN AI Agent 导航系统"))
        self._print_line()
        print(f" 场景: {self._c(Colors.CYAN, scene_name)}")
        print(f" 目标: {self._c(Colors.GREEN, goal)}")
        print(f" 模式: {self._c(Colors.YELLOW, mode)}")
        self._print_separator()
        self._start_time = datetime.now()

    def print_semantic_phase(self, info: Dict[str, Any]):
        """打印语义地图定位阶段信息"""
        print()
        print(self._c(Colors.BOLD + Colors.WHITE,
                      "┌─ 第一步：语义地图定位 " + "─" * 45))
        print()

        if info.get("semantic_available", False):
            print(f"  {self._c(Colors.CYAN, '[语义解析]')} 正在解析 .semantic.txt ...")
            print(f"  {self._c(Colors.CYAN, '[语义解析]')} 找到 {info.get('num_categories', 0)} 个语义类别")
            print(f"  {self._c(Colors.CYAN, '[语义解析]')} {info.get('goal_room', '')}特征物体匹配:")

            for match in info.get("matches", []):
                status = "✓" if match.get("found") else " "
                color = Colors.GREEN if match.get("found") else Colors.GRAY
                match_name = match["name"]
                print(f"              {self._c(color, f'{status} {match_name}')}")

            print()
            print(f"  {self._c(Colors.BLUE, '[语义定位]')} {info.get('goal_room', '')}区域坐标范围:")
            print(f"              中心: ({info['center_x']:.2f}, {info['center_z']:.2f})")
            print(f"              范围: 宽 {info['width']:.1f}m × 深 {info['depth']:.1f}m")

            print()
            print(f"  {self._c(Colors.GREEN, '[路径规划]')} 当前位置 → {info.get('goal_room', '')}区域")
            print(f"              距离: {info['distance']:.1f} 米")
            print(f"              路点: {info['num_waypoints']} 个导航点")
            print(f"              预计步数: ~{info['estimated_steps']} 步")

            print()
            print(f"  {self._c(Colors.GREEN, '[导航模式]')} 使用语义定位模式（有语义标签可用）")
        else:
            print(f"  {self._c(Colors.YELLOW, '⚠ [提示]')} 未找到 .semantic.txt 语义标签文件")
            print(f"     {self._c(Colors.YELLOW, '[处理]')} 自动降级为纯 AI 视觉导航模式")
            print(f"     {self._c(Colors.YELLOW, '[影响]')} 导航效率可能降低，但功能不受影响")

        print()
        print(self._c(Colors.GRAY, "└" + "─" * 66))

    def print_navigation_step(
        self,
        step: int,
        total_steps: int,
        remaining_distance: float,
        elapsed_seconds: float,
        perception_info: Dict[str, Any],
        ai_info: Dict[str, Any],
        execution_info: Dict[str, Any],
        status_info: Dict[str, Any],
    ):
        """打印导航循环中每一步的完整信息"""
        # 计算已用时间
        elapsed = str(timedelta(seconds=int(elapsed_seconds)))

        # 步骤标题
        self._print_line()
        print(
            f"  步骤 {step}/{total_steps}  "
            f"|  剩余距离: {remaining_distance:.1f}m  "
            f"|  已用时间: {elapsed}"
        )
        self._print_separator()
        print()

        # 感知信息
        print(f"  {self._c(Colors.CYAN, '[感知]')} 正在捕获第一视角图片 ...")
        if perception_info.get("original_size"):
            orig = perception_info["original_size"]
            comp = perception_info.get("compressed_size", "?")
            print(f"  {self._c(Colors.CYAN, '[感知]')} 图片尺寸: {orig[0]}×{orig[1]} → 压缩至 {comp}")
        if perception_info.get("file_size_kb"):
            print(f"  {self._c(Colors.CYAN, '[感知]')} 图片大小: {perception_info['file_size_kb']:.0f} KB")
        if perception_info.get("api_sent"):
            print(f"  {self._c(Colors.CYAN, '[感知]')} 正在发送至 {perception_info.get('model_name', 'AI')} API ...")
        print()

        # AI 推理信息
        print(f"  {self._c(Colors.YELLOW, '[AI 推理]')} {self._c(Colors.YELLOW, '─' * 50)}")
        print(f"    场景描述: \"{self._c(Colors.WHITE, ai_info.get('scene_description', ''))}\"")
        print(f"    推理过程: \"{self._c(Colors.WHITE, ai_info.get('reasoning', ''))}\"")
        action = ai_info.get("action", "unknown")
        action_color = Colors.GREEN if action == "arrived" else Colors.YELLOW
        action_display = f"{action} ✓" if action == "arrived" else action
        print(f"    决策指令: {self._c(action_color, action_display)}")
        print(f"  {self._c(Colors.YELLOW, '─' * 62)}")
        print()

        # 执行信息
        print(f"  {self._c(Colors.GREEN, '[执行]')} 动作: {execution_info.get('action', 'unknown')}")
        if execution_info.get("displacement") is not None:
            print(f"  {self._c(Colors.GREEN, '[执行]')} 位移: {execution_info['displacement']:.2f}m")
        if execution_info.get("new_position"):
            pos = execution_info["new_position"]
            print(f"  {self._c(Colors.GREEN, '[执行]')} 新位置: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        if execution_info.get("new_heading"):
            print(f"  {self._c(Colors.GREEN, '[执行]')} 新朝向: {execution_info['new_heading']}")
        if execution_info.get("position_unchanged"):
            print(f"  {self._c(Colors.GREEN, '[执行]')} 位置未变（转向动作）")
        print()

        # 状态信息
        progress = status_info.get("progress", 0)
        remaining = status_info.get("remaining_steps", 0)
        print(f"  {self._c(Colors.BLUE, '[状态]')} 进度: {progress:.1f}%  |  剩余步数: ~{remaining} 步")
        if status_info.get("is_stuck", False):
            print(f"  {self._c(Colors.BG_YELLOW + Colors.BLACK, ' ⚠ [警告] ')} {self._c(Colors.YELLOW, '检测到卡住！')}")
            print(f"     {self._c(Colors.YELLOW, '[原因]')} 连续 {status_info.get('stuck_count', 0)} 步执行 move_forward 但位置未变化")
            print(f"     {self._c(Colors.YELLOW, '[处理]')} 强制执行 turn_left，尝试新的方向")
            print(f"     {self._c(Colors.YELLOW, '[建议]')} 前方可能有障碍物，已自动绕行")
        if status_info.get("api_error"):
            print(f"  {self._c(Colors.RED, '✗ [错误]')} API 调用失败")
            print(f"     {self._c(Colors.RED, '[原因]')} {status_info.get('api_error_reason', '未知错误')}")
            print(f"     {self._c(Colors.RED, '[处理]')} 第 {status_info.get('retry_count', 1)} 次重试 (最多 {status_info.get('max_retries', 3)} 次) ...")
            if status_info.get("retry_success"):
                print(f"     {self._c(Colors.GREEN, '[结果]')} 重试成功，继续导航")

        print()
        self._print_separator()

    def print_arrival(self, stats: Dict[str, Any]):
        """打印到达目标时的信息"""
        self._print_line()
        print(f"  步骤 {stats['step']}/{stats['total_steps']}  "
              f"|  剩余距离: {stats['remaining_distance']:.1f}m  "
              f"|  已用时间: {str(timedelta(seconds=int(stats['elapsed_seconds'])))}")
        self._print_separator()
        print()
        print(f"  {self._c(Colors.CYAN, '[感知]')} 正在捕获第一视角图片 ...")
        print(f"  {self._c(Colors.CYAN, '[感知]')} 正在发送至 AI API ...")
        print()
        print(f"  {self._c(Colors.YELLOW, '[AI 推理]')} {self._c(Colors.YELLOW, '─' * 50)}")
        print(f"    场景描述: \"{self._c(Colors.WHITE, stats.get('scene_description', ''))}\"")
        print(f"    推理过程: \"{self._c(Colors.WHITE, stats.get('reasoning', ''))}\"")
        print(f"    决策指令: {self._c(Colors.GREEN, 'arrived ✓')}")
        print(f"  {self._c(Colors.YELLOW, '─' * 62)}")
        print()
        print(f"  {self._c(Colors.GREEN, '[执行]')} 动作: arrived — 已到达目标！")
        print()
        print(f"  {self._c(Colors.GREEN, '✅ 导航成功！')}")
        print(f"     目标: {stats.get('goal', '')}")
        print(f"     总步数: {stats['step']} 步")
        print(f"     总距离: {stats.get('total_distance', 0):.1f} 米")
        print(f"     总用时: {str(timedelta(seconds=int(stats['elapsed_seconds'])))}")
        print(f"     API 调用次数: {stats.get('api_calls', 0)} 次")

    def print_summary(self, stats: Dict[str, Any]):
        """打印导航结束时的完整统计总结"""
        print()
        self._print_line()
        print(f"  {self._c(Colors.GREEN, '🎯 导航完成！')}")
        self._print_line()
        print()
        print(f"  目标:          {stats.get('goal', '')}")
        result_color = Colors.GREEN if stats.get('success', False) else Colors.RED
        result_text = "✅ 成功到达" if stats.get('success', False) else "❌ 未到达"
        print(f"  结果:          {self._c(result_color, result_text)}")
        print()
        print(f"  统计信息:")
        print(f"  {self._c(Colors.GRAY, '┌' + '─' * 53 + '┐')}")
        self._print_stat_row("总步数", f"{stats.get('step', 0)} / {stats.get('total_steps', 0)} 步")
        self._print_stat_row("总距离", f"{stats.get('total_distance', 0):.1f} 米")
        self._print_stat_row("总用时", str(timedelta(seconds=int(stats.get('elapsed_seconds', 0)))))
        avg_speed = stats.get('step', 0) / max(stats.get('elapsed_seconds', 1), 1)
        self._print_stat_row("平均步速", f"{avg_speed:.2f} 步/秒")
        self._print_stat_row("API 调用次数", f"{stats.get('api_calls', 0)} 次")
        self._print_stat_row("总 Token 消耗", f"{stats.get('total_tokens', 0):,} tokens")
        self._print_stat_row("预估 API 成本", f"${stats.get('total_cost', 0):.4f}")
        self._print_stat_row("转向次数", f"{stats.get('turn_count', 0)} 次")
        self._print_stat_row("卡住恢复次数", f"{stats.get('stuck_recovery_count', 0)} 次")
        print(f"  {self._c(Colors.GRAY, '└' + '─' * 53 + '┘')}")
        print()

        # 动作统计
        action_counts = stats.get('action_counts', {})
        total_actions = sum(action_counts.values()) or 1
        print(f"  动作统计:")
        print(f"  {self._c(Colors.GRAY, '┌' + '─' * 53 + '┐')}")
        for action in ["move_forward", "turn_left", "turn_right", "arrived"]:
            count = action_counts.get(action, 0)
            pct = count / total_actions * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  │ {action:<20} {count:>4} 次 ({pct:>5.1f}%) {bar} │")
        print(f"  {self._c(Colors.GRAY, '└' + '─' * 53 + '┘')}")
        print()

        # 输出文件
        output_files = stats.get('output_files', {})
        if output_files:
            print(f"  输出文件:")
            for key, path in output_files.items():
                print(f"  ├─ {key}: {self._c(Colors.CYAN, path)}")

        print()
        self._print_line()

    def print_warning(self, message: str, details: Optional[Dict[str, str]] = None):
        """打印警告信息"""
        print(f"  {self._c(Colors.BG_YELLOW + Colors.BLACK, ' ⚠ [警告] ')} {self._c(Colors.YELLOW, message)}")
        if details:
            for key, value in details.items():
                print(f"     {self._c(Colors.YELLOW, f'[{key}]')} {value}")

    def print_error(self, message: str, details: Optional[Dict[str, str]] = None):
        """打印错误信息"""
        print(f"  {self._c(Colors.RED, '✗ [错误]')} {message}")
        if details:
            for key, value in details.items():
                print(f"     {self._c(Colors.RED, f'[{key}]')} {value}")

    def print_info(self, tag: str, message: str, color: str = Colors.CYAN):
        """打印一般信息"""
        print(f"  {self._c(color, f'[{tag}]')} {message}")

    def print_user_prompt(self):
        """打印用户输入提示"""
        print()
        print(self._c(Colors.BOLD + Colors.GREEN, "请输入导航目标（例如：去客厅、找到厨房、走到卧室）"))
        print(self._c(Colors.GRAY, "输入 'q' 退出，输入 'help' 查看帮助"))
        print(self._c(Colors.GREEN, "> "), end="", flush=True)

    def _print_line(self):
        """打印分隔线"""
        print(self._c(Colors.GRAY, "═" * 70))

    def _print_separator(self):
        """打印虚线分隔"""
        print(self._c(Colors.GRAY, "─" * 70))

    def _print_stat_row(self, label: str, value: str):
        """打印统计行"""
        print(f"  {self._c(Colors.GRAY, '│')} {label:<20} {value:<30} {self._c(Colors.GRAY, '│')}")
