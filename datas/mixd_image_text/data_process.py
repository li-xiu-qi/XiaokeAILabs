from markdown_image_enhancer import enhance_markdown_images


if __name__ == "__main__":
    # 测试示例
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # 测试用的Markdown文本
    file_name = "./web_datas/全球公认的三大帅气、高颜值狗，有你家的爱犬吗？.md"
    with open(file_name,encoding="utf-8") as f:
        test_markdown = f.read()
    
    # 从环境变量获取配置
    api_key = os.getenv("ZHIPU_API_KEY")
    base_url = os.getenv("ZHIPU_BASE_URL")
    vision_model = os.getenv("ZHIPU_MODEL")
    
    print("🚀 开始测试Markdown图片增强器...")
    print("原始Markdown:")
    print(test_markdown)
    print("\n" + "="*50 + "\n")
    
    # 测试增强功能
    enhanced = enhance_markdown_images(
        test_markdown,
        provider="zhipu",
        api_key=api_key,
        base_url=base_url,
        vision_model=vision_model
    )
    
    print("增强后的Markdown:")
    print(enhanced)
    # 写入增强后的Markdown到文件
    with open(file_name.replace(".md","_images.md"), "w", encoding="utf-8") as f:
        f.write(enhanced)
