import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from threading import Thread
from typing import List, Dict
import argparse

# ----------------- 模型路径 -----------------
MODEL_PATHS = {
    "Qwen1.5": r"E:\moodel\ma\qianwen\qwen\Qwen1___5-0___5B-Chat",
    "Qwen2.5": r"E:\moodel\ma\qianwen\qwen\Qwen2___5-0___5B-Instruct",
    "Qwen3": r"E:\moodel\ma\qianwen\qwen\Qwen3-0___6B",
}

# ----------------- 模型缓存 -----------------
_loaded_models = {}  # 缓存已加载模型

def load_model(model_name: str):
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    model_path = MODEL_PATHS.get(model_name, model_name)  # 如果自定义路径则直接用输入
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        tie_word_embeddings=False
    ).eval()
    model.generation_config.max_new_tokens = 1024

    _loaded_models[model_name] = (model, tokenizer)
    return model, tokenizer

# ----------------- 流式对话 -----------------
def chat_stream(model_name: str, query: str, history: List[Dict]):
    model, tokenizer = load_model(model_name)

    conversation = [{"role": "system", "content": "You are a helpful assistant."}]
    for query_h, response_h in history:
        conversation.append({"role": "user", "content": query_h})
        conversation.append({"role": "assistant", "content": response_h})
    conversation.append({"role": "user", "content": query})

    inputs_dict = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt")
    input_ids = inputs_dict["input_ids"].to(model.device)
    attention_mask = inputs_dict.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True)
    generation_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, streamer=streamer, max_new_tokens=512)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text

# ----------------- Gradio 回调 -----------------
def respond(model_name: str, message: str, chat_history: List[Dict]):
    # 添加用户消息
    chat_history.append({"role": "user", "content": message})
    full_response = ""
    for new_text in chat_stream(model_name, message, chat_history[:-1]):
        full_response += new_text
        # 更新最后一条助手消息
        if len(chat_history) > 1 and chat_history[-1]["role"] == "assistant":
            chat_history[-1]["content"] = full_response
        else:
            chat_history.append({"role": "assistant", "content": full_response})
        yield chat_history

def reset_state(chat_history):
    chat_history.clear()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return chat_history

# ----------------- 主函数 -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", default=False, help="创建可分享链接")
    parser.add_argument("--inbrowser", action="store_true", default=False, help="自动在浏览器打开")
    parser.add_argument("--server-port", type=int, default=8000, help="端口号")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="服务器名")
    args = parser.parse_args()

    with gr.Blocks() as demo:
        gr.Markdown("<h1 align='center'>Multi-Qwen ChatBot</h1>")

        model_selector = gr.Dropdown(
            choices=list(MODEL_PATHS.keys()),
            value="Qwen1.5",
            label="选择模型",
            allow_custom_value=True  # ✅ 支持自定义模型名
        )

        chatbot = gr.Chatbot(label="ChatBot")
        user_input = gr.Textbox(lines=2, label="输入消息")
        chat_history = gr.State([])

        with gr.Row():
            submit_btn = gr.Button("发送")
            reset_btn = gr.Button("清除历史")

        submit_btn.click(respond, [model_selector, user_input, chat_history], [chatbot], show_progress=True)
        reset_btn.click(reset_state, [chat_history], [chatbot])
        submit_btn.click(lambda: gr.update(value=""), [], [user_input])

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name
    )

if __name__ == "__main__":
    main()