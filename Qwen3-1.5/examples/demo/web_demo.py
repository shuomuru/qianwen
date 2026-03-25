# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio."""

from argparse import ArgumentParser
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

DEFAULT_CKPT_PATH = r'E:\moodel\ma\qianwen\qwen\Qwen1___5-0___5B-Chat'

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--share", action="store_true", default=False)
    parser.add_argument("--inbrowser", action="store_true", default=True)
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    return parser.parse_args()


# ✅ 加载模型（修复 resume_download 问题）
def _load_model_tokenizer(args):
    print("正在加载模型...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        local_files_only=True
    )

    device_map = "cpu" if args.cpu_only else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        torch_dtype="auto",
        device_map=device_map,
        local_files_only=True
    ).eval()

    print("模型加载完成！")
    return model, tokenizer


# ✅ 核心聊天逻辑（修复 generate bug）
def _chat_stream(model, tokenizer, query, history):
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    for q, a in history:
        conversation.append({"role": "user", "content": q})
        conversation.append({"role": "assistant", "content": a})

    conversation.append({"role": "user", "content": query})

    # 构造输入
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # 流式输出
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        input_ids=inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ✅ Web UI
def _launch_demo(args, model, tokenizer):

    def predict(_query, _chatbot, _task_history):
        _chatbot.append((_query, ""))
        response = ""

        for new_text in _chat_stream(model, tokenizer, _query, _task_history):
            response += new_text
            _chatbot[-1] = (_query, response)
            yield _chatbot

        _task_history.append((_query, response))

    def regenerate(_chatbot, _task_history):
        if not _task_history:
            yield _chatbot
            return
        last = _task_history.pop(-1)
        _chatbot.pop(-1)
        yield from predict(last[0], _chatbot, _task_history)

    def reset_user_input():
        return gr.update(value="")

    def reset_state(_chatbot, _task_history):
        _task_history.clear()
        _chatbot.clear()
        _gc()
        return _chatbot

    with gr.Blocks() as demo:
        gr.Markdown("# 🤖 Qwen1.5 ChatBot（本地版）")

        chatbot = gr.Chatbot(label="Qwen Chat")
        query = gr.Textbox(lines=2, label="输入问题")
        task_history = gr.State([])

        with gr.Row():
            submit_btn = gr.Button("发送")
            regen_btn = gr.Button("重试")
            clear_btn = gr.Button("清空")

        submit_btn.click(predict, [query, chatbot, task_history], [chatbot])
        submit_btn.click(reset_user_input, [], [query])

        regen_btn.click(regenerate, [chatbot, task_history], [chatbot])
        clear_btn.click(reset_state, [chatbot, task_history], [chatbot])

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()
    model, tokenizer = _load_model_tokenizer(args)
    _launch_demo(args, model, tokenizer)


if __name__ == "__main__":
    main()