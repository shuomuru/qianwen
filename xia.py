from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-0.6B',cache_dir='./')#cache_dir='./'为指定下载目录，不然会下载到C盘的缓存中