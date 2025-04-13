from DriverAssistanceSystem import DriverAssistanceSystem

if __name__ == "__main__":
    print("启动驾驶辅助系统...")

    try:
        system = DriverAssistanceSystem()
        system.run()
    except Exception as e:
        print(f"系统启动失败: {e}")
        import traceback

        traceback.print_exc()

    print("系统已关闭")