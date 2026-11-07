"""Validate the one-image-per-row Flickr split contract without ML dependencies."""
import re


def clean_captions(captions):
    """Require a caption list and discard blank strings before pairing."""
    if not isinstance(captions, (list, tuple)):
        raise ValueError('caption 必须是字符串列表')
    if any(not isinstance(caption, str) for caption in captions):
        raise ValueError('caption 列表只能包含字符串')
    cleaned = [caption.strip() for caption in captions if caption.strip()]
    if not cleaned:
        raise ValueError('图片至少需要一条非空描述')
    return cleaned


def partition_indices(splits, image_ids):
    if len(splits) != len(image_ids):
        raise ValueError('split 与 img_id 数量不一致')
    partitions = {'train': [], 'val': [], 'test': []}
    owners = {}
    for index, (split, image_id) in enumerate(zip(splits, image_ids)):
        if split not in partitions:
            raise ValueError(f'未知数据划分: {split!r}')
        if image_id is None or str(image_id).strip() == '':
            raise ValueError(f'第 {index} 条记录缺少 img_id')
        raw = str(image_id).strip()
        if not re.fullmatch(r'[+-]?\d+', raw):
            raise ValueError(f'无效的 img_id: {image_id!r}')
        key = str(int(raw))
        if key in owners:
            if owners[key] != split:
                raise ValueError(f'同一图片 ID 跨数据划分: {key}')
            raise ValueError(f'同一图片 ID 在划分中重复: {key}')
        owners[key] = split
        partitions[split].append(index)
    if any(not indices for indices in partitions.values()):
        raise ValueError('train、val、test 三个划分都必须非空')
    return partitions
