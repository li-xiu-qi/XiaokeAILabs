#!/bin/bash
source ~/xinfer-env/bin/activate

echo "=== 读取 Q4_K_XL GGUF 元数据 ==="
python3 << 'EOF'
import struct, sys

def read_gguf_metadata(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            return {"error": "not a GGUF file"}

        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_kv_count = struct.unpack('<Q', f.read(8))[0]

        # 简单读取一些元数据
        metadata = {"version": version, "tensor_count": tensor_count, "metadata_kv_count": metadata_kv_count}

        # 读取 metadata
        for _ in range(metadata_kv_count):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')

            value_type = struct.unpack('<I', f.read(4))[0]
            if value_type == 8:  # string
                val_len = struct.unpack('<Q', f.read(8))[0]
                value = f.read(val_len).decode('utf-8')
            elif value_type == 10:  # uint64
                value = struct.unpack('<Q', f.read(8))[0]
            elif value_type == 6:  # int32
                value = struct.unpack('<i', f.read(4))[0]
            elif value_type == 4:  # uint32
                value = struct.unpack('<I', f.read(4))[0]
            else:
                break

            metadata[key] = value

        return metadata

gguf_path = "/home/ke/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf"
meta = read_gguf_metadata(gguf_path)
print(json.dumps(meta, indent=2, ensure_ascii=False))
EOF
