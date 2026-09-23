# FastMCP 项目专用客户端示例 - 全面测试服务器功能
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # 启动 MCP 工具服务器连接
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],  # 修正文件名
    )
    server_connection = stdio_client(server_params)
    read, write = await server_connection.__aenter__()
    mcp_session = ClientSession(read, write)
    await mcp_session.__aenter__()
    await mcp_session.initialize()

    print("🚀 MCP 服务器连接成功！")
    print("=" * 50)

    # 1. 测试获取可用工具
    print("\n📋 1. 获取可用工具:")
    tools = await mcp_session.list_tools()
    if tools.tools:
        for tool in tools.tools:
            print(f"  - {tool.name}: {tool.description}")
    else:
        print("  没有可用工具")

    # 2. 测试调用工具
    print("\n🔧 2. 测试工具调用:")
    if tools.tools:
        # 测试计算工具
        for tool in tools.tools:
            if tool.name == "calculate":
                print(f"  调用 {tool.name} 工具:")
                try:
                    # 测试简单数学运算
                    result = await mcp_session.call_tool("calculate", {"expression": "2 + 3 * 4"})
                    print(f"    计算 '2 + 3 * 4': {result.content[0].text}")
                    
                    # 测试复杂运算
                    result = await mcp_session.call_tool("calculate", {"expression": "(10 + 5) / 3"})
                    print(f"    计算 '(10 + 5) / 3': {result.content[0].text}")
                except Exception as e:
                    print(f"    调用失败: {e}")
    else:
        print("  没有工具可测试")

    # 3. 测试获取可用提示
    print("\n💡 3. 获取可用提示:")
    try:
        prompts = await mcp_session.list_prompts()
        if prompts.prompts:
            for prompt in prompts.prompts:
                print(f"  - {prompt.name}: {prompt.description}")
        else:
            print("  没有可用提示")
    except Exception as e:
        print(f"  获取提示失败: {e}")

    # 4. 测试调用提示
    print("\n📝 4. 测试提示调用:")
    try:
        prompts = await mcp_session.list_prompts()
        if prompts.prompts:
            # 测试 ask_about_topic 提示
            for prompt in prompts.prompts:
                if prompt.name == "ask_about_topic":
                    print(f"  调用 {prompt.name} 提示:")
                    result = await mcp_session.get_prompt("ask_about_topic", {"topic": "人工智能"})
                    print(f"    生成的提示: {result.messages[0].content.text}")
                    
                elif prompt.name == "generate_code_request":
                    print(f"  调用 {prompt.name} 提示:")
                    result = await mcp_session.get_prompt("generate_code_request", 
                                                        {"language": "Python", 
                                                         "task_description": "排序一个列表"})
                    print(f"    生成的提示: {result.messages[0].content.text}")
        else:
            print("  没有提示可测试")
    except Exception as e:
        print(f"  调用提示失败: {e}")

    # 5. 测试获取可用资源
    print("\n📂 5. 获取可用资源:")
    try:
        resources = await mcp_session.list_resources()
        if resources.resources:
            for resource in resources.resources:
                print(f"  - {resource.uri}: {resource.name}")
        else:
            print("  没有可用资源")
    except Exception as e:
        print(f"  获取资源失败: {e}")

    # 6. 测试读取资源
    print("\n📖 6. 测试资源读取:")
    try:
        resources = await mcp_session.list_resources()
        if resources.resources:
            for resource in resources.resources:
                print(f"  读取资源 {resource.uri}:")
                try:
                    result = await mcp_session.read_resource(resource.uri)
                    print(f"    内容: {result.contents[0].text}")
                except Exception as e:
                    print(f"    读取失败: {e}")
        else:
            print("  没有资源可读取")
    except Exception as e:
        print(f"  读取资源失败: {e}")

    print("\n" + "=" * 50)
    print("✅ 所有测试完成！")

    # 清理连接
    await mcp_session.__aexit__(None, None, None)
    await server_connection.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(main())

