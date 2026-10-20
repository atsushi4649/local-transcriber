// 推論はすべてこのワーカー内（端末上）で実行する。音声・テキストは外部に送信しない。
import {
  pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer, InterruptableStoppingCriteria, env,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowLocalModels = false;
env.useBrowserCache = true; // モデルは Cache Storage に保存され、2回目以降はオフラインで読み込まれる

const ASR_MODELS = {
  'base-ja': 'onnx-community/moonshine-base-ja-ONNX',
  // tiny-ja は decoder_model_merged が無く transformers.js 3.8.1 で読み込めないため除外
};
// 要約モデル。webllm: GPU 専用（WebLLM）、tjs: CPU でも動く（transformers.js）
// {q} は計算精度（q4f16_1 / q4f32_1）に置き換える
const LLM_MODELS = {
  'qwen3.5-2b': { engine: 'webllm', id: 'Qwen3.5-2B-{q}-MLC' },
  'gemma2-2b-jpn': { engine: 'webllm', id: 'gemma-2-2b-jpn-it-{q}-MLC', noSystem: true },
  'qwen3.5-4b': { engine: 'webllm', id: 'Qwen3.5-4B-{q}-MLC' },
  'qwen3.5-0.8b': { engine: 'webllm', id: 'Qwen3.5-0.8B-{q}-MLC' },
  'qwen3-0.6b-cpu': { engine: 'tjs', id: 'onnx-community/Qwen3-0.6B-ONNX' },
};

let device = 'wasm';
let f16 = false;
let mobileGpu = false; // Qualcomm / ARM などのモバイル GPU（半精度で出力が壊れやすい）
let asr = null, asrKey = null;
let llm = null, llmKey = null;

// WebGPU が使えるか調べる。使えない理由も診断表示用に返す。
async function detectDevice() {
  let gpuDiag;
  try {
    if (!navigator.gpu) gpuDiag = 'navigator.gpu なし（ワーカー内で WebGPU が無効）';
    else {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) gpuDiag = 'requestAdapter() が null（ブラウザが GPU を許可していない）';
      else {
        device = 'webgpu'; f16 = adapter.features.has('shader-f16');
        const i = adapter.info || {};
        gpuDiag = `OK: ${i.vendor || '?'} ${i.architecture || ''} ${i.description || ''}`.trim();
        mobileGpu = /qualcomm|adreno|arm|mali|imagination|powervr|samsung/i.test(`${i.vendor} ${i.architecture} ${i.description}`)
          || /Android/i.test(navigator.userAgent);
      }
    }
  } catch (e) { gpuDiag = 'エラー: ' + (e?.message || e); }
  return { device, f16, mobileGpu, gpuDiag };
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

// precision: 'auto' | 'f32' | 'f16'。自動ではモバイル GPU や f16 非対応 GPU で f32 を使う
function quant(precision) {
  const useF32 = precision === 'f32' || (precision !== 'f16' && (mobileGpu || !f16));
  return useF32 ? 'q4f32_1' : 'q4f16_1';
}

async function loadLlm(name, precision) {
  const base = LLM_MODELS[name] || LLM_MODELS['qwen3.5-2b'];
  const conf = { ...base, id: base.id.replace('{q}', quant(precision)) };
  if (llm && llmKey === conf.id) return llm;
  if (conf.engine === 'webllm' && device !== 'webgpu') {
    throw new Error('この端末では GPU（WebGPU）が使えないため、このモデルは動きません。設定で「Qwen3 0.6B（CPU）」を選んでください。');
  }
  await unloadLlm(); // 切り替え時は前のモデルを必ず解放する（メモリ不足エラー対策）
  post({ type: 'status', text: '要約モデルを読み込み中…（初回は1〜2GB 程度のダウンロード）' });
  llm = conf.engine === 'webllm' ? await loadWebLlm(conf) : await loadTjsLlm(conf);
  llmKey = conf.id;
  post({ type: 'llm-loaded', id: conf.id });
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
        messages, stream: true, temperature: 0.3, top_p: 0.9, max_tokens: 700,
        repetition_penalty: 1.1, frequency_penalty: 0.4, extra_body: { enable_thinking: false },
      });
      let text = '';
      interrupt = () => engine.interruptGenerate();
      try {
        for await (const chunk of stream) {
          const t = chunk.choices[0]?.delta?.content || '';
          if (!t) continue;
          text += t;
          if (looping(text)) { engine.interruptGenerate(); break; }
          if (garbled(text)) { engine.interruptGenerate(); throw new Error(GARBLED_MSG); }
          onToken?.(clean(text));
        }
      } finally { interrupt = null; }
      return clean(text);
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
      const stopper = new InterruptableStoppingCriteria();
      interrupt = () => stopper.interrupt();
      let text = '';
      const streamer = new TextStreamer(tokenizer, {
        skip_prompt: true, skip_special_tokens: true,
        callback_function: (t) => {
          text += t;
          if (looping(text)) stopper.interrupt(); else onToken?.(clean(text));
        },
      });
      try {
        await model.generate({
          ...inputs, max_new_tokens: 700, do_sample: false, repetition_penalty: 1.15,
          streamer, stopping_criteria: stopper,
        });
      } finally { interrupt = null; }
      return clean(text);
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

// 生成中の現在のモデルを止める関数（停止ボタン・繰り返し検知用）
let interrupt = null;

// GPU の計算が壊れると、無関係な文字体系（アラビア文字・ベンガル文字・タイ文字・キリル文字・
// ハングルなど）や中国語・英単語の羅列になり、ひらがながほぼ出なくなる。
// 60 文字以上出た時点で、そのどちらかに当てはまれば文字化けとみなす。
const GARBLED_MSG = '出力が文字化けしています（GPU の計算が正しく行われていません）。'
  + '設定の「GPU の計算精度」を「f32（互換）」にして、もう一度要約してください。';
function garbled(text) {
  const t = text.replace(/\s/g, '');
  if (t.length < 60) return false;
  const count = (re) => t.match(re)?.length || 0;
  const foreign = count(/[Ѐ-ӿ֐-ۿऀ-෿฀-๿가-힯]/g);
  const hiragana = count(/[ぁ-ゟ]/g);
  const latin = count(/[A-Za-z]/g);
  // 箇条書きだけの要約はひらがなが少ないことがあるので、英字だらけの場合に限って判定する
  return foreign / t.length > 0.1 || (hiragana / t.length < 0.05 && latin / t.length > 0.25);
}

// 小さいモデルが同じ行や同じフレーズを繰り返し始めたら true
function looping(text) {
  const lines = text.split('\n').map((l) => l.trim()).filter((l) => l.length >= 4);
  const count = new Map();
  for (const l of lines) {
    const n = (count.get(l) || 0) + 1;
    if (n >= 3) return true;
    count.set(l, n);
  }
  const tail = text.slice(-300);
  for (let len = 6; len <= 100; len++) {
    const x = tail.slice(-len);
    if (x.trim() && tail.endsWith(x + x + x)) return true;
  }
  return false;
}

// 思考タグを除き、同じ行の2回目以降を消す（空行は残す）
function clean(text) {
  const seen = new Set();
  return stripThink(text).split('\n').filter((l) => {
    const k = l.trim();
    if (!k) return true;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  }).join('\n').trim();
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

const SYS = 'あなたは日本語の文字起こしを要約するアシスタントです。文字起こしには誤認識が含まれることがあるので、文脈から自然に補って理解してください。事実を付け加えず、内容に忠実に、簡潔な日本語で書いてください。これからの予定を過去のこととして書かないでください。';

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

async function summarize({ text, llmModel, precision, style }) {
  stopRequested = false;
  const m = await loadLlm(llmModel, precision);
  const format = style === 'minutes'
    ? '議事録としてまとめてください。見出し「## 概要」の下に全体を2〜3文の文章で、見出し「## 要点」の下に重要な内容を箇条書きで、見出し「## 決定事項・TODO」の下に、決まったことや誰が何をいつまでにするかを箇条書きで書いてください。各見出しは1回だけ使い、最後まで書いたら終えてください。'
    : '要約してください。見出し「## 概要」の下に全体を2〜3文の文章で、見出し「## 要点」の下に重要な内容を3〜7個の箇条書きで書いてください。各見出しは1回だけ使い、最後まで書いたら終えてください。';

  let source = text;
  const parts = splitText(text);
  if (parts.length > 1) {
    // 長い文字起こしは分割して要点を抽出してから、最後にまとめる
    const notes = [];
    for (let i = 0; i < parts.length && !stopRequested; i++) {
      post({ type: 'status', text: `長文のため分割して処理中… (${i + 1}/${parts.length})` });
      const n = await generate(m, [
        { role: 'system', content: SYS },
        { role: 'user', content: `【文字起こしの一部】\n${parts[i]}\n\n上の内容の重要な点を、箇条書きで5個以内にまとめてください。` },
      ]);
      notes.push(n);
    }
    source = notes.join('\n');
  }
  if (stopRequested) { post({ type: 'summary-done', text: '', stopped: true }); return; }
  post({ type: 'status', text: '要約を生成中…' });
  post({ type: 'summary-reset' });
  const final = await generate(m, [
    { role: 'system', content: SYS },
    { role: 'user', content: `【${parts.length > 1 ? '長い文字起こしから抽出したメモ' : '文字起こし'}】\n${source}\n\n上の内容を${format}` },
  ], (t) => post({ type: 'summary-token', text: t }));
  post({ type: 'summary-done', text: final, stopped: stopRequested });
}

// メッセージを1件ずつ順番に処理する
let queue = Promise.resolve();
let stopRequested = false;
self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === 'stop') { stopRequested = true; interrupt?.(); return; }
  queue = queue.then(async () => {
    try {
      if (msg.type === 'init') post({ type: 'ready', ...(await detectDevice()) });
      else if (msg.type === 'transcribe') await transcribe(msg);
      else if (msg.type === 'summarize') await summarize(msg);
      else if (msg.type === 'preload') {
        await loadAsr(msg.asrModel);
        await loadLlm(msg.llmModel, msg.precision);
        post({ type: 'preloaded' });
      }
      post({ type: 'idle', for: msg.type, id: msg.id });
    } catch (err) {
      console.error(err);
      post({ type: 'error', for: msg.type, id: msg.id, text: String(err?.message || err) });
    }
  });
};
