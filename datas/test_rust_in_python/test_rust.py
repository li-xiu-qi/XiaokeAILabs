def test_rust_adder():
    try:
        from rust_adder import add_numbers

        # 测试加法函数
        result = add_numbers(5, 7)
        print(f"Rust加法结果: 5 + 7 = {result}")
        assert result == 12, "加法结果不正确"

        print("测试成功通过!")
    except ImportError:
        print("无法导入Rust模块。请确保你已经构建了Rust库。")
        print("请执行: cd datas/test_rust_in_python && maturin develop")


if __name__ == "__main__":
    test_rust_adder()
