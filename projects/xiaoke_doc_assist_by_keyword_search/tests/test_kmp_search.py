from typing import List

from projects.xiaoke_doc_assist_by_keyword_search.utils.kmp_search import KMPSearch


def main():
    """
    主函数，测试KMP搜索算法
    """
    # 创建搜索实例
    kmp = KMPSearch()

    # 测试用例
    text: str = "这是一个测试文本这是一个测试文本"
    pattern: str = "测试"

    # 执行搜索
    result: List[int] = kmp.search(text, pattern)

    # 输出结果
    print(f"文本: {text}")
    print(f"关键词: {pattern}")
    print(f"匹配位置: {result}")
    print(f"总计找到 {len(result)} 个匹配")

    # 显示每个匹配的具体位置
    for pos in result:
        print(f"找到匹配于位置 {pos}: {text[pos:pos + len(pattern)]}")


if __name__ == "__main__":
    main()