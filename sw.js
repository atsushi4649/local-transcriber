// ページ本体とライブラリをキャッシュし、オフラインでも起動できるようにする。
// （モデル本体は transformers.js が別のキャッシュに保存する）
const CACHE = 'app-v6';
const CORE = ['./', 'index.html', 'worker.js', 'recorder-worklet.js', 'manifest.webmanifest', 'icon.svg'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(CORE)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys()
    .then((ks) => Promise.all(ks.filter((k) => k.startsWith('app-') && k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

// COOP/COEP ヘッダーを付けて crossOriginIsolated にし、WASM のマルチスレッドを有効にする
// （GitHub Pages などヘッダーを設定できないホスティング向け）
function isolate(res) {
  if (res.status === 0 || res.type === 'opaqueredirect') return res;
  const h = new Headers(res.headers);
  h.set('Cross-Origin-Opener-Policy', 'same-origin');
  h.set('Cross-Origin-Embedder-Policy', 'credentialless');
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
}

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET') return;
  const sameOrigin = url.origin === location.origin;
  const lib = url.hostname === 'cdn.jsdelivr.net';
  if (!sameOrigin && !lib) return; // モデル（huggingface）は transformers.js 側のキャッシュに任せる
  e.respondWith((async () => {
    const cache = await caches.open(CACHE);
    if (sameOrigin) {
      // 自サイトのファイル: ネット優先（更新を反映）、オフライン時はキャッシュ
      let res;
      try {
        res = await fetch(e.request);
        if (res.ok) cache.put(e.request, res.clone());
      } catch {
        res = await cache.match(e.request, { ignoreSearch: true });
        if (!res) return Response.error();
      }
      return isolate(res);
    }
    // ライブラリ（バージョン固定）: キャッシュ優先
    const hit = await cache.match(e.request);
    if (hit) return hit;
    const res = await fetch(e.request);
    if (res.ok) cache.put(e.request, res.clone());
    return res;
  })());
});
