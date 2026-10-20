// 推論はすべてこのワーカー内（端末上）で実行する。音声・テキストは外部に送信しない。
import {
  pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer, env,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowLocalModels = false;
env.useBrowserCache = true; // モデルは Cache Storage に保存され、2回目以降はオフラインで読み込まれる

const ASR_MODELS = {
  'base-ja': 'onnx-community/moonshine-base-ja-ONNX',
  // tiny-ja は decoder_model_merged が無く transformers.js 3.8.1 で読み込めないため除外
};
// 要約モデル。webllm: GPU 専用（WebLLM）、tjs: CPU でも動く（transformers.js）
const LLM_MODELS = {
  'qwen3.5-2b': { engine: 'webllm', id: 'Qwen3.5-2B-q4f16_1-MLC' },
  'gemma2-2b-jpn': { engine: 'webllm', id: 'gemma-2-2b-jpn-it-q4f16_1-MLC', noSystem: true },
  'qwen3.5-4b': { engine: 'webllm', id: 'Qwen3.5-4B-q4f16_1-MLC' },
  'qwen3.5-0.8b': { engine: 'webllm', id: 'Qwen3.5-0.8B-q4f16_1-MLC' },
  'qwen3-0.6b-cpu': { engine: 'tjs', id: 'onnx-community/Qwen3-0.6B-ONNX' },
};

let device = 'wasm';
let f16 = false;
let asr = null, asrKey = null;
let llm = null, llmKey = null;

async function detectDevice() {
  try {
    if (navigator.gpu) {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) { device = 'webgpu'; f16 = adapter.features.has('shader-f16'); }
    }
  } catch { /* WASM にフォールバック */ }
  return { device, f16 };
}

function progress(kind) {
  return (p) => {
    if (p.status === 'progress') post({ type: 'progress', kind, file: p.file, loaded: p.loaded, total: p.total });
    else if (p.status === 'done') post({ type: 'progress', kind, file: p.file, done: true });
  };
}

function post(msg) { self.postMessage(msg); }

async function loadAsr(key) {
  if (!ASR_MODELS[key]) key = 'base-ja';
  if (asr && asrKey === key) return asr;
  if (asr) { await asr.dispose?.(); asr = null; }
  post({ type: 'status', text: '文字起こしモデルを読み込み中…' });
  // Moonshine は WebGPU だと出力が壊れる（3.8.1 で確認）ため CPU で動かす。
  // エンコーダ fp32 の方が句読点まで正しく出る。
  asr = await pipeline('automatic-speech-recognition', ASR_MODELS[key], {
    device: 'wasm',
    dtype: { encoder_model: 'fp32', decoder_model_merged: 'q8' },
    progress_callback: progress('asr'),
  });
  // transformers.js 3.8.1 では日本語 Moonshine の processor に tokenizer が渡らず
  // "Unable to decode without a tokenizer" になるので、手動で渡す
  if (asr.processor && !asr.processor.tokenizer) asr.processor.components.tokenizer = asr.tokenizer;
  asrKey = key;
  return asr;
}

async function unloadLlm() {
  if (!llm) return;
  try { await llm.unload(); } catch (e) { console.warn(e); }
  llm = null; llmKey = null;
}

async function loadLlm(name) {
  if (llm && llmKey === name) return llm;
  const conf = LLM_MODELS[name] || LLM_MODELS['qwen3.5-2b'];
  if (conf.engine === 'webllm' && device !== 'webgpu') {
    throw new Error('この端末では GPU（WebGPU）が使えないため、このモデルは動きません。設定で「Qwen3 0.6B（CPU）」を選んでください。');
  }
  await unloadLlm(); // 切り替え時は前のモデルを必ず解放する（メモリ不足エラー対策）
  post({ type: 'status', text: '要約モデルを読み込み中…（初回は1〜2GB 程度のダウンロード）' });
  llm = conf.engine === 'webllm' ? await loadWebLlm(conf) : await loadTjsLlm(conf);
  llmKey = name;
  return llm;
}

let webllmEngine = null;
async function loadWebLlm(conf) {
  const webllm = await import('https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.85/+esm');
  const onProgress = (r) => post({ type: 'llm-progress', progress: r.progress, text: r.text });
  if (!webllmEngine) {
    webllmEngine = await webllm.CreateMLCEngine(conf.id, { initProgressCallback: onProgress });
  } else {
    webllmEngine.setInitProgressCallback(onProgress);
    await webllmEngine.reload(conf.id);
  }
  post({ type: 'llm-progress', progress: 1 });
  const engine = webllmEngine;
  return {
    async chat(messages, onToken) {
      if (conf.noSystem) messages = mergeSystem(messages);
      const stream = await engine.chat.completions.create({
        messages, stream: true, temperature: 0.2, top_p: 0.9, max_tokens: 900,
        frequency_penalty: 0.3, extra_body: { enable_thinking: false },
      });
      let text = '';
      for await (const chunk of stream) {
        const t = chunk.choices[0]?.delta?.content || '';
        if (t) { text += t; onToken?.(stripThink(text), true); }
      }
      return stripThink(text);
    },
    async unload() { await engine.unload(); },
  };
}

async function loadTjsLlm(conf) {
  const tokenizer = await AutoTokenizer.from_pretrained(conf.id, { progress_callback: progress('llm') });
  const model = await AutoModelForCausalLM.from_pretrained(conf.id, {
    device: 'wasm', dtype: 'int8', progress_callback: progress('llm'),
  });
  return {
    async chat(messages, onToken) {
      const inputs = tokenizer.apply_chat_template(messages, {
        add_generation_prompt: true, return_dict: true, enable_thinking: false,
      });
      let text = '';
      const streamer = new TextStreamer(tokenizer, {
        skip_prompt: true, skip_special_tokens: true,
        callback_function: (t) => { text += t; onToken?.(stripThink(text), true); },
      });
      await model.generate({ ...inputs, max_new_tokens: 900, do_sample: false, repetition_penalty: 1.1, streamer });
      return stripThink(text);
    },
    async unload() { await model.dispose(); },
  };
}

// system ロール非対応のモデル（Gemma）向けに、指示を最初の user メッセージへまとめる
function mergeSystem(messages) {
  const sys = messages.filter((m) => m.role === 'system').map((m) => m.content).join('\n');
  const rest = messages.filter((m) => m.role !== 'system');
  if (sys && rest[0]) rest[0] = { ...rest[0], content: sys + '\n\n' + rest[0].content };
  return rest;
}

function stripThink(t) {
  return t.replace(/<think>[\s\S]*?(<\/think>|$)/g, '').trimStart();
}

async function transcribe({ id, audio, asrModel }) {
  const p = await loadAsr(asrModel);
  const sec = audio.length / 16000;
  // 日本語は1秒あたりのトークン数が多いので上限を広めに取る（暴走防止のため上限あり）
  const max_new_tokens = Math.min(448, Math.ceil(sec * 15) + 10);
  const out = await p(audio, { max_new_tokens });
  post({ type: 'segment', id, text: cleanup(out.text) });
}

// 1文字ごとに入る空白を除去（英単語どうしの間の空白だけ残す）し、
// 同じフレーズの異常な繰り返し（ASR のよくある誤動作）を抑える
function cleanup(t) {
  t = (t || '').trim().replace(/\s+/g, ' ');
  t = t.replace(/ (?![A-Za-z])|(?<![A-Za-z]) /g, '');
  return t.replace(/(.{2,20}?)\1{3,}/g, '$1');
}

async function generate(m, messages, onToken) {
  return (await m.chat(messages, onToken)).trim();
}

const SYS = 'あなたは日本語の文字起こしを要約するアシスタントです。文字起こしには誤認識が含まれることがあるので、文脈から自然に補って理解してください。事実を付け加えず、内容に忠実に、簡潔な日本語で書いてください。予定や決定事項の時制（〜する予定、〜までに行う等）は元の発言のまま保ってください。';

function splitText(text, size = 2200) {
  const parts = [];
  let buf = '';
  for (const s of text.split(/(?<=[。！？\n])/)) {
    if ((buf + s).length > size && buf) { parts.push(buf); buf = ''; }
    buf += s;
  }
  if (buf.trim()) parts.push(buf);
  return parts;
}

async function summarize({ text, llmModel, style }) {
  const m = await loadLlm(llmModel);
  const format = style === 'minutes'
    ? '次の形式で出力してください。\n## 概要\n（2〜3文）\n## 要点\n- …\n## 決定事項・TODO\n- 担当者：やること（期限）\n\n「決定事項・TODO」には、文字起こしに出てきた担当・期限・日程をすべて、上の1行の形で具体的に書いてください。担当者が不明なら「担当未定」と書いてください。'
    : '次の形式で出力してください。\n## 概要\n（2〜3文）\n## 要点\n- …（3〜7個）';

  let source = text;
  const parts = splitText(text);
  if (parts.length > 1) {
    // 長い文字起こしは分割して要点を抽出してから、最後にまとめる
    const notes = [];
    for (let i = 0; i < parts.length; i++) {
      post({ type: 'status', text: `長文のため分割して処理中… (${i + 1}/${parts.length})` });
      const n = await generate(m, [
        { role: 'system', content: SYS },
        { role: 'user', content: `以下は文字起こしの一部です。重要な点を箇条書きで5個以内にまとめてください。\n\n${parts[i]}` },
      ]);
      notes.push(n);
    }
    source = notes.join('\n');
  }
  post({ type: 'status', text: '要約を生成中…' });
  post({ type: 'summary-reset' });
  const final = await generate(m, [
    { role: 'system', content: SYS },
    { role: 'user', content: `${parts.length > 1 ? '以下は長い文字起こしから抽出した要点です。全体の' : '以下の文字起こしの'}要約を作ってください。\n${format}\n\n---\n${source}` },
  ], (t) => post({ type: 'summary-token', text: t }));
  post({ type: 'summary-done', text: final });
}

// メッセージを1件ずつ順番に処理する
let queue = Promise.resolve();
self.onmessage = (e) => {
  const msg = e.data;
  queue = queue.then(async () => {
    try {
      if (msg.type === 'init') post({ type: 'ready', ...(await detectDevice()) });
      else if (msg.type === 'transcribe') await transcribe(msg);
      else if (msg.type === 'summarize') await summarize(msg);
      else if (msg.type === 'preload') {
        await loadAsr(msg.asrModel);
        await loadLlm(msg.llmModel);
        post({ type: 'preloaded' });
      }
      post({ type: 'idle', for: msg.type, id: msg.id });
    } catch (err) {
      console.error(err);
      post({ type: 'error', for: msg.type, id: msg.id, text: String(err?.message || err) });
    }
  });
};
