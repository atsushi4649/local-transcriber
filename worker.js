// 推論はすべてこのワーカー内（端末上）で実行する。音声・テキストは外部に送信しない。
import {
  pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer, env,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowLocalModels = false;
env.useBrowserCache = true; // モデルは Cache Storage に保存され、2回目以降はオフラインで読み込まれる

const ASR_MODELS = {
  'base-ja': 'onnx-community/moonshine-base-ja-ONNX',
  'tiny-ja': 'onnx-community/moonshine-tiny-ja-ONNX',
};
// ext: 重みが .onnx_data に分かれている形式（use_external_data_format が必要）
const LLM_MODELS = {
  'qwen3-0.6b': { id: 'onnx-community/Qwen3-0.6B-ONNX' },
  'gemma3-1b': { id: 'onnx-community/gemma-3-1b-it-ONNX', ext: true },
  'lfm2-1.2b': { id: 'onnx-community/LFM2-1.2B-ONNX', ext: true },
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

async function loadLlm(name, dev) {
  const useGpu = dev !== 'cpu' && device === 'webgpu';
  const key = name + (useGpu ? '@gpu' : '@cpu');
  if (llm && llmKey === key) return llm;
  if (llm) { await llm.model.dispose?.(); llm = null; }
  post({ type: 'status', text: '要約モデルを読み込み中…（初回は数百MB〜1GB超のダウンロード）' });
  const { id, ext } = LLM_MODELS[name];
  const tokenizer = await AutoTokenizer.from_pretrained(id, { progress_callback: progress('llm') });
  const model = await AutoModelForCausalLM.from_pretrained(id, {
    device: useGpu ? 'webgpu' : 'wasm',
    dtype: useGpu ? (f16 ? 'q4f16' : 'q4') : (ext ? 'q4' : 'int8'),
    ...(ext ? { use_external_data_format: true } : {}),
    progress_callback: progress('llm'),
  });
  llm = { tokenizer, model };
  llmKey = key;
  return llm;
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

async function generate({ tokenizer, model }, messages, onToken) {
  const inputs = tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true, return_dict: true, enable_thinking: false,
  });
  let inThink = false;
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true, skip_special_tokens: true,
    callback_function: (t) => {
      if (t.includes('<think>')) inThink = true;
      if (!inThink) onToken?.(t);
      if (t.includes('</think>')) inThink = false;
    },
  });
  const out = await model.generate({
    ...inputs, max_new_tokens: 900, do_sample: false, repetition_penalty: 1.1, streamer,
  });
  const len = inputs.input_ids.dims[1];
  const text = tokenizer.batch_decode(out.slice(null, [len, null]), { skip_special_tokens: true })[0];
  return text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
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

async function summarize({ text, llmModel, llmDevice, style }) {
  const m = await loadLlm(llmModel, llmDevice);
  const format = style === 'minutes'
    ? '次の形式で出力してください。\n## 概要\n（2〜3文）\n## 要点\n- …\n## 決定事項・TODO\n- …（なければ「なし」）'
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
        await loadLlm(msg.llmModel, msg.llmDevice);
        post({ type: 'preloaded' });
      }
      post({ type: 'idle', for: msg.type, id: msg.id });
    } catch (err) {
      console.error(err);
      post({ type: 'error', for: msg.type, id: msg.id, text: String(err?.message || err) });
    }
  });
};
