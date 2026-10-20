# ローカル文字起こし

スマホ・タブレットのブラウザだけで動く、日本語の文字起こし＋要約アプリです。
音声もテキストも端末の外に送信しません（初回にモデルをダウンロードした後は機内モードでも動きます）。

- 文字起こし: [Moonshine](https://github.com/moonshine-ai/moonshine-v2) Base / Tiny 日本語（CPU で実行）
- 要約: Qwen3 0.6B ほか（WebGPU、または CPU で実行）
- 実行環境: [Transformers.js](https://github.com/huggingface/transformers.js) 3.8.1

日本語 Moonshine モデルは Moonshine Community License（非商用）です。
