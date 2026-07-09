#!/usr/bin/env python3
"""
VLN AI Agent 导航演示入口
交互式 VLN 导航系统主程序

用法：
    python run_vln_demo.py
    python run_vln_demo.py --scene "path/to/scene.glb" --goal "去客厅"

首次运行需要设置 OpenAI API Key：
    set OPENAI_API_KEY=sk-your-key-here
    或在程序提示时输入
"""

import argparse
import logging
import os
import sys
import time

# 必须在导入 habitat 之前执行路径设置
import setup_path

import numpy as np

from dataset_selector import discover_datasets, discover_scenes, resolve_assets
from vln_agent.config import VLNConfig, default_config
from vln_agent.agent import VLNAgent
from vln_agent.navigation_loop import NavigationLoop
from vln_agent.instruction_parser import parse_instruction, get_help_text
from vln_agent.terminal_printer import TerminalPrinter, Colors

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_api_key(config: VLNConfig) -> bool:
    """
    设置 API Key
    优先从环境变量读取，如果没有则提示用户输入

    Returns:
        是否成功设置
    """
    config.load_from_env()

    if config.is_openai_configured():
        return True

    printer = TerminalPrinter()
    print()
    printer.print_info("配置", "未检测到 OpenAI API Key", Colors.YELLOW)
    print()
    print("请选择 API Key 设置方式：")
    print("  1. 输入 API Key（仅本次会话有效）")
    print("  2. 设置环境变量后重启程序（推荐）")
    print("     set OPENAI_API_KEY=sk-your-key-here")
    print("  3. 退出程序")
    print()

    choice = input("请选择 (1/2/3): ").strip()

    if choice == "1":
        api_key = input("请输入你的 OpenAI API Key: ").strip()
        if api_key:
            config.api.openai_api_key = api_key
            os.environ["OPENAI_API_KEY"] = api_key
            print(f"\n{Colors.GREEN}✓ API Key 已设置{Colors.RESET}")
            return True
        else:
            print(f"\n{Colors.RED}✗ API Key 不能为空{Colors.RESET}")
            return False
    elif choice == "2":
        print(f"\n{Colors.YELLOW}请设置环境变量后重新运行程序：{Colors.RESET}")
        print("  set OPENAI_API_KEY=sk-your-key-here")
        return False
    else:
        print(f"\n{Colors.GRAY}退出程序{Colors.RESET}")
        return False


def select_scene_interactive() -> tuple:
    """
    交互式选择场景

    Returns:
        (scene_path, navmesh_path, semantic_txt_path, scene_name)
    """
    printer = TerminalPrinter()

    print()
    printer.print_info("数据", "正在扫描数据集...", Colors.CYAN)

    # 发现数据集
    datasets = discover_datasets()
    if not datasets:
        printer.print_error("未找到任何数据集", {
            "提示": "请确保 HM3D 数据集已下载到 data/scene_datasets/ 目录"
        })
        sys.exit(1)

    # 选择数据集
    print(f"\n{Colors.BOLD}可用数据集:{Colors.RESET}")
    for i, ds_path in enumerate(datasets):
        # 统计场景数量
        scene_count = len(discover_scenes(ds_path))
        print(f"  {i + 1}. {ds_path.name} ({scene_count} 个场景)")

    ds_choice = int(input("\n请选择数据集编号: ").strip()) - 1
    if ds_choice < 0 or ds_choice >= len(datasets):
        printer.print_error("无效的选择")
        sys.exit(1)

    selected_dataset_path = datasets[ds_choice]

    # 发现场景
    scenes = discover_scenes(selected_dataset_path)
    if not scenes:
        printer.print_error(f"数据集 {selected_dataset_path.name} 中没有找到场景文件")
        sys.exit(1)

    # 选择场景
    print(f"\n{Colors.BOLD}可用场景 (显示前20个):{Colors.RESET}")
    for i, scene_path in enumerate(scenes[:20]):
        # 显示场景名称（使用父目录名/文件名）
        scene_display = f"{scene_path.parent.name}/{scene_path.name}"
        print(f"  {i + 1}. {scene_display}")
    if len(scenes) > 20:
        print(f"  ... 还有 {len(scenes) - 20} 个场景")

    sc_choice = int(input("\n请选择场景编号: ").strip()) - 1
    if sc_choice < 0 or sc_choice >= len(scenes):
        printer.print_error("无效的选择")
        sys.exit(1)

    selected_scene_path = scenes[sc_choice]

    # 解析配套资产（返回 SceneConfig 对象）
    scene_config = resolve_assets(selected_scene_path)

    return (
        str(scene_config.scene_path),
        str(scene_config.navmesh_path) if scene_config.navmesh_path else "",
        str(scene_config.semantic_txt_path) if scene_config.semantic_txt_path else "",
        selected_scene_path.stem,
    )


def create_simulator(scene_path: str, navmesh_path: str):
    """
    创建 Habitat 仿真器

    Args:
        scene_path: 场景文件路径
        navmesh_path: 导航网格文件路径

    Returns:
        Habitat 仿真器实例
    """
    # 设置环境变量，强制使用 CPU/OSMesa 渲染（解决 WSL 中无 CUDA/EGL 的问题）
    import os
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    os.environ.setdefault("GLOG_minloglevel", "2")

    try:
        import habitat_sim
        import habitat_sim.utils.common as utils
    except ImportError:
        print(f"{Colors.RED}✗ 导入 habitat_sim 失败{Colors.RESET}")
        print("请确保 Habitat-Sim 已正确安装")
        sys.exit(1)

    # 配置仿真器
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = -1  # 强制使用 CPU，禁用 GPU


    # 配置 RGB 相机
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "rgb"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [1440, 1920]  # [height, width]
    rgb_sensor_spec.hfov = 75
    rgb_sensor_spec.position = [0.0, 0.88, 0.0]

    # 配置代理
    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor_spec]

    # 创建仿真器
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    # 加载导航网格
    if navmesh_path and os.path.exists(navmesh_path):
        sim.pathfinder.load_nav_mesh(navmesh_path)
        print(f"{Colors.GREEN}✓ 导航网格已加载{Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}⚠ 未找到导航网格文件，路径规划可能受限{Colors.RESET}")

    # 在导航网格上放置智能体
    _place_agent_on_navmesh(sim)

    return sim


def _place_agent_on_navmesh(sim):
    """在导航网格上随机放置智能体"""
    import habitat_sim
    import numpy as np

    if not sim.pathfinder.is_loaded:
        return

    # 在可导航区域随机采样
    for _ in range(100):
        point = sim.pathfinder.get_random_navigable_point()
        if point is not None:
            agent = sim.get_agent(0)
            agent_state = habitat_sim.AgentState()
            agent_state.position = point
            agent_state.rotation = [0.0, 0.0, 0.0, 1.0]
            agent.set_state(agent_state)
            print(f"{Colors.GREEN}✓ 智能体已放置在: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}){Colors.RESET}")
            return

    print(f"{Colors.YELLOW}⚠ 无法在导航网格上放置智能体{Colors.RESET}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VLN AI Agent 导航演示")
    parser.add_argument("--scene", type=str, help="场景文件路径")
    parser.add_argument("--navmesh", type=str, help="导航网格文件路径")
    parser.add_argument("--semantic", type=str, help="语义标签文件路径")
    parser.add_argument("--goal", type=str, help="导航目标（如：去客厅）")
    parser.add_argument("--no-color", action="store_true", help="禁用 ANSI 颜色输出")
    parser.add_argument("--max-steps", type=int, default=300, help="最大步数")
    parser.add_argument("--model", type=str, default="gpt-4o", help="AI 模型名称")
    args = parser.parse_args()

    printer = TerminalPrinter(use_color=not args.no_color)

    # 显示欢迎信息
    print()
    print(f"{Colors.BOLD}{Colors.WHITE}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.WHITE}  Habitat-VLN AI Agent 导航系统 v1.0{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.WHITE}{'=' * 70}{Colors.RESET}")
    print()
    print(f"  基于 GPT-4o 视觉语言模型的智能室内导航")
    print(f"  支持自然语言指令：\"去客厅\"、\"找到厨房\"、\"走到卧室\"")
    print()

    # 配置
    config = default_config
    # 从 config.json 加载配置
    config.load_from_json()
    if args.max_steps:
        config.navigation.max_steps = args.max_steps
    if args.model:
        config.api.openai_model = args.model

    # 设置 API Key
    if not setup_api_key(config):
        sys.exit(1)

    # 选择场景
    if args.scene:
        scene_path = args.scene
        navmesh_path = args.navmesh or ""
        semantic_txt_path = args.semantic or ""
        scene_name = os.path.basename(scene_path)
    else:
        scene_path, navmesh_path, semantic_txt_path, scene_name = select_scene_interactive()

    # 验证场景文件
    if not os.path.exists(scene_path):
        printer.print_error(f"场景文件不存在: {scene_path}")
        sys.exit(1)

    # 创建仿真器
    print()
    printer.print_info("仿真", "正在创建 Habitat 仿真环境...", Colors.CYAN)
    sim = create_simulator(scene_path, navmesh_path)

    try:
        # 交互式导航循环
        while True:
            # 获取用户目标
            if args.goal:
                goal = args.goal
                args.goal = None  # 只在第一次使用
            else:
                printer.print_user_prompt()
                goal = sys.stdin.readline().strip()

            # 处理特殊命令
            if goal.lower() in ("q", "quit", "exit", "退出"):
                print(f"\n{Colors.GREEN}感谢使用 VLN 导航系统！{Colors.RESET}")
                break

            if goal.lower() in ("help", "h", "帮助"):
                print(get_help_text())
                continue

            if not goal:
                continue

            # 解析指令
            parsed = parse_instruction(goal)
            if parsed is None:
                printer.print_warning(
                    f"无法理解指令: '{goal}'",
                    {"提示": "请尝试更明确的指令，如 '去客厅'、'找到厨房'。输入 'help' 查看帮助"}
                )
                continue

            printer.print_info(
                "解析",
                f"指令解析成功: 目标={parsed['goal_room']}, 置信度={parsed['confidence']:.0%}",
                Colors.GREEN,
            )

            # 初始化 Agent
            agent = VLNAgent(config)
            mode = agent.initialize(
                sim=sim,
                scene_name=scene_name,
                goal=goal,
                semantic_txt_path=semantic_txt_path if os.path.exists(semantic_txt_path) else None,
            )

            # 运行导航循环
            nav_loop = NavigationLoop(agent, config)
            stats = nav_loop.run()

            # 清理
            agent.cleanup()

            # 询问是否继续
            print()
            print(f"{Colors.GRAY}─{Colors.RESET}" * 70)
            print(f"{Colors.BOLD}是否继续导航到其他位置？{Colors.RESET}")
            print(f"  输入新的目标继续，或输入 'q' 退出")
            print()

    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}用户中断导航{Colors.RESET}")
    except Exception as e:
        printer.print_error(f"程序异常", {"错误": str(e)})
        logger.exception("程序异常")
    finally:
        # 清理仿真器
        try:
            sim.close()
            print(f"{Colors.GRAY}仿真器已关闭{Colors.RESET}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
