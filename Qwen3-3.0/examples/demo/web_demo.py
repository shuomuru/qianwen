# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on Gradio for Qwen3."""

from argparse import ArgumentParser
from threading import Thread
import gc

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

DEFAULT_CKPT_PATH = r"E:\moodel\ma\qianwen\qwen\Qwen3-0___6B"


def _get_args():
    parser = ArgumentParser(description="Qwen3 web chat demo.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default=DEFAULT_CKPT_PATH,
        help="Checkpoint path, default to %(default)r",
    )
    parser.add_argument("--cpu-only", action="store_true", help="Run demo on CPU only")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--inbrowser", action="store_true", help="Open browser automatically")
    parser.add_argument("--server-port", type=int, default=8000, help="Server port")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Server name")
    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    """加载 Qwen3 模型与 tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)

    device_map = "cpu" if args.cpu_only else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        torch_dtype="auto",
        device_map=device_map,
    ).eval()

    model.generation_config.max_new_tokens = 2048
    return model, tokenizer


def _chat_stream(model, tokenizer, query, history):
    """流式生成 Qwen3 回复"""
    conversation = history.copy()
    conversation.append({"role": "user", "content": query})

    input_text = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )

    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True
    )

    thread = Thread(target=model.generate, kwargs={**inputs, "streamer": streamer})
    thread.start()

    for new_text in streamer:
        yield new_text


def _launch_demo(args, model, tokenizer):
    def predict(_query, _chatbot, _task_history):
        """处理用户输入并流式返回 Gradio Chatbot"""
        user_msg = {"role": "user", "content": _query}
        _task_history.append(user_msg)
        _chatbot.append(user_msg)

        response_text = ""
        for new_text in _chat_stream(model, tokenizer, _query, history=_task_history):
            response_text += new_text
            if _chatbot[-1].get("role") == "assistant":
                _chatbot[-1]["content"] = response_text
            else:
                _chatbot.append({"role": "assistant", "content": response_text})
            yield _chatbot

        _task_history.append({"role": "assistant", "content": response_text})
        yield _chatbot

    def regenerate(_chatbot, _task_history):
        """重试最后一条消息"""
        if len(_task_history) < 2:
            yield _chatbot
            return
        _task_history.pop(-1)  # 删除 assistant
        last_user = _task_history.pop(-1)  # 删除 user
        _chatbot.clear()
        _chatbot.extend(_task_history)
        yield from predict(last_user["content"], _chatbot, _task_history)

    def reset_user_input():
        return gr.update(value="")

    def reset_state(_chatbot, _task_history):
        _task_history.clear()
        _chatbot.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _chatbot

    with gr.Blocks() as demo:
        gr.Markdown(
            "<p align='center'><img src='https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/assets/logo/qwen2.5_logo.png' style='height:120px'/></p>"
        )
        gr.Markdown(
            "<center><font size=3>This WebUI is based on Qwen3, developed by Alibaba Cloud.</font></center>"
        )

        chatbot = gr.Chatbot(label="Qwen", elem_classes="control-height")
        query = gr.Textbox(lines=2, label="Input")
        task_history = gr.State([])

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History")
            submit_btn = gr.Button("🚀 Submit")
            regen_btn = gr.Button("🤔 Regenerate")

        submit_btn.click(
            predict, [query, chatbot, task_history], [chatbot], show_progress=True
        )
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(reset_state, [chatbot, task_history], outputs=[chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)

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