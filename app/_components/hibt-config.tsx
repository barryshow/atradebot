"use client";
import { useState, useEffect } from "react";

interface HibtConfig {
  token: string;
  authorization: string;
  xAuthToken: string;
  bgetKey: string;
  bgetId: string;
  feishuWebhook: string;
}

const STORAGE_KEY = "atradebot_hibt_config";

export function HibtConfigPanel() {
  const [cfg, setCfg] = useState<HibtConfig>({
    token: "", authorization: "", xAuthToken: "",
    bgetKey: "", bgetId: "", feishuWebhook: "",
  });
  const [saved, setSaved] = useState(false);
  const [show, setShow] = useState(false);
  const [pasteMode, setPasteMode] = useState<"json" | "manual">("json");
  const [rawJson, setRawJson] = useState("");

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try { setCfg(JSON.parse(stored)); } catch {}
    }
  }, []);

  const handlePaste = () => {
    try {
      const parsed = JSON.parse(rawJson);
      const newCfg = { ...cfg };
      if (parsed.token) newCfg.token = parsed.token;
      if (parsed.authorization) newCfg.authorization = parsed.authorization;
      if (parsed.xAuthToken) newCfg.xAuthToken = parsed.xAuthToken;
      if (parsed.bgetKey) newCfg.bgetKey = parsed.bgetKey;
      if (parsed.bgetId) newCfg.bgetId = parsed.bgetId;
      if (parsed.bget_key) newCfg.bgetKey = parsed.bget_key;
      if (parsed.bget_id) newCfg.bgetId = parsed.bget_id;
      if (parsed.xAuthToken || parsed["x-auth-token"]) newCfg.xAuthToken = parsed.xAuthToken || parsed["x-auth-token"];
      setCfg(newCfg);
      setRawJson("");
    } catch {
      alert("JSON格式错误，请检查粘贴内容");
    }
  };

  const saveToBackend = async () => {
    try {
      await fetch("/api/engine/config/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cfg),
      });
    } catch {}
  };

  const handleSave = () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(cfg));
    saveToBackend();
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  if (!show) {
    return (
      <button
        onClick={() => setShow(true)}
        className="text-xs text-gray-500 hover:text-gray-300 underline"
      >
        HIBT Token配置
      </button>
    );
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6 w-full max-w-lg max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-bold text-white">HIBT Token 配置</h2>
          <button onClick={() => setShow(false)} className="text-gray-400 hover:text-white text-xl">&times;</button>
        </div>

        {/* 粘贴模式切换 */}
        <div className="flex gap-2 mb-4">
          <button
            onClick={() => setPasteMode("json")}
            className={`px-3 py-1 text-xs rounded ${pasteMode === "json" ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-400"}`}
          >
            粘贴JSON
          </button>
          <button
            onClick={() => setPasteMode("manual")}
            className={`px-3 py-1 text-xs rounded ${pasteMode === "manual" ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-400"}`}
          >
            手动输入
          </button>
        </div>

        {pasteMode === "json" ? (
          <div className="space-y-3">
            <p className="text-xs text-gray-400">
              粘贴从浏览器F12复制的完整Headers JSON
            </p>
            <textarea
              className="w-full h-32 bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-gray-200 font-mono"
              placeholder='{"token":"eyJ...","authorization":"eyJ...","xAuthToken":"eyJ...","bgetKey":"xxx","bgetId":"xxx"}'
              value={rawJson}
              onChange={(e) => setRawJson(e.target.value)}
            />
            <button
              onClick={handlePaste}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded"
            >
              解析并填充
            </button>
          </div>
        ) : (
          <div className="space-y-3">
            {[
              { key: "token", label: "Token / HIBT_TOKEN", ph: "eyJ..." },
              { key: "authorization", label: "Authorization (可选)", ph: "eyJ..." },
              { key: "xAuthToken", label: "x-auth-token (可选)", ph: "eyJ..." },
              { key: "bgetKey", label: "bget_key / vKey (可选)", ph: "HotsCoin..." },
              { key: "bgetId", label: "bget_id / memberId (可选)", ph: "211892" },
            ].map(({ key, label, ph }) => (
              <div key={key}>
                <label className="text-xs text-gray-400 block mb-1">{label}</label>
                <input
                  className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-gray-200 font-mono"
                  placeholder={ph}
                  value={(cfg as any)[key] || ""}
                  onChange={(e) => setCfg({ ...cfg, [key]: e.target.value })}
                />
              </div>
            ))}
          </div>
        )}

        <div className="mt-4 pt-4 border-t border-gray-700 flex gap-3">
          <button
            onClick={handleSave}
            className="px-6 py-2 bg-green-600 hover:bg-green-500 text-white rounded font-medium"
          >
            {saved ? "✔️ 已保存" : "保存配置"}
          </button>
          <button
            onClick={() => {
              setCfg({ token: "", authorization: "", xAuthToken: "", bgetKey: "", bgetId: "", feishuWebhook: "" });
              localStorage.removeItem(STORAGE_KEY);
            }}
            className="px-4 py-2 bg-red-600 hover:bg-red-500 text-white rounded text-sm"
          >
            清除
          </button>
          <span className="text-xs text-gray-500 self-center">
            保存后请重启引擎生效
          </span>
        </div>

        {(cfg.token || cfg.authorization) && (
          <div className="mt-3 text-xs text-green-400">
            ✅ Token已配置 {cfg.token.slice(0, 20)}...
          </div>
        )}
      </div>
    </div>
  );
}