# 运行迟分测试的简单脚本
import sys
import os

# 添加项目根目录到路径
project_root = r"C:\Users\k\Documents\project\programming_project\python_project\importance\XiaokeAILabs"
sys.path.append(project_root)

# 运行测试
if __name__ == "__main__":
    print("开始运行迟分测试...")
    
    try:
        from datas.test_late_chunking.test_late_chunking import test_late_chunking, compare_with_traditional_chunking
    
        
        print("\n" + "="*60)
        print("运行比较测试")
        print("="*60)
        
        # 运行比较测试
        compare_with_traditional_chunking()
        
    except Exception as e:
        print(f"运行测试时出现错误: {e}")
        import traceback
        traceback.print_exc()
