# Multi-Qwen ChatBot

Multi-Qwen ChatBot 是一个基于 Qwen 系列大语言模型的多模型对话系统，支持流式输出和自定义模型选择。用户可以通过 Web 界面与不同 Qwen 模型实时交互，进行自然语言问答或对话生成。

该项目展示了大语言模型加载、缓存、流式生成以及多模型管理的完整流程，适合作为深度学习项目展示或面试项目。

---

## 功能特性

- **多模型支持**：可选择 Qwen1.5、Qwen2.5、Qwen3，支持自定义模型路径。
- **流式输出**：使用 `TextIteratorStreamer` 实现生成文本实时显示。
- **历史记录管理**：自动记录对话历史，支持一键清除。
- **自定义输入清理**：过滤异常字符，提高输出稳定性。
- **本地/公网运行**：可通过 Gradio WebUI 访问，可生成公网分享链接。

---

## 安装与运行

### 克隆项目
```bash
git clone https://github.com/QwenLM/Qwen1.5.git

⚠️ 注意：项目已更新为 Qwen3，可能无法直接获取旧版本模型，建议手动下载或在分支中查看其他版本。

安装依赖
# 安装 Python 环境
conda create -n Qwen1.5 python=3.8
conda activate Qwen1.5

# 安装核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.32
pip install accelerate>=0.26.0
pip install modelscope
下载模型权重
from modelscope import snapshot_download

model_dir = snapshot_download('模型权重名称', cache_dir='./')  # 指定下载目录
运行项目
python run1.py --server-port 8000 --inbrowser
--share：生成公网访问链接
--inbrowser：启动后自动打开浏览器
访问界面
http://127.0.0.1:8000
使用说明
选择模型：下拉菜单中选择 Qwen1.5、Qwen2.5、Qwen3 或输入自定义模型路径。
输入消息：在文本框中输入对话内容。
发送消息：点击“发送”，模型实时生成回复。
清除历史：点击“清除历史”，重置对话。
技术实现
模型加载与缓存：避免重复加载，支持 GPU/CPU 自动选择。
流式生成：利用 TextIteratorStreamer + 多线程实时返回模型输出，提升用户体验。
多模型管理：使用 MODEL_PATHS 字典维护多个模型路径，可灵活扩展新模型。
Gradio WebUI：可视化界面，支持实时交互与多模型选择。
项目亮点
简洁易用的多模型对话系统
支持自定义模型，兼容多种 Qwen 版本
流式显示生成文本，用户体验良好
可直接作为面试项目或演示项目使用
未来改进方向
优化在使用 Qwen2.5-0.5B 模型时回答乱码的问题
增加上下文记忆，跨会话保持对话历史
增强异常字符处理，进一步提升生成稳定性
增加日志与统计，方便分析模型表现
支持多用户并发访问，构建在线服务平台
