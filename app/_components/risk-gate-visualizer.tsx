"use client";
import type { RiskGateResult } from "@/lib/types/engine";

interface GateNode {
  level: number;
  name: string;
  icon: string;
}

// 实际使用的风控门（不再显示已移除的死区/震荡门）
const GATE_NODES: GateNode[] = [
  { level: 1, name: "ML概率(0.30)", icon: "🧠" },
  { level: 2, name: "极值翻转", icon: "🔄" },
  { level: 2, name: "共振参考", icon: "📊" },
  { level: 3, name: "AI分析(仅展示)", icon: "🤖" },
];

function getGateStatus(
  node: GateNode,
  gates: RiskGateResult[]
): "idle" | "pass" | "fail" | "flip" {
  const match = gates.filter((g) => g.level === node.level);
  if (match.length === 0) return "idle";
  if (node.name.includes("极值翻转")) {
    const flipped = match.some((g) => g.reason?.includes("翻转") || g.name.includes("flip"));
    return flipped ? "flip" : "pass";
  }
  const last = match[match.length - 1];
  return last.passed ? "pass" : "fail";
}

export function RiskGateVisualizer({ gates }: { gates: RiskGateResult[] }) {
  return (
    <div className="flex items-center gap-1 overflow-x-auto py-2">
      {GATE_NODES.map((node, i) => {
        const status = getGateStatus(node, gates);
        const colors =
          status === "pass"
            ? "bg-green-900/60 border-green-700 text-green-300"
            : status === "fail"
            ? "bg-red-900/60 border-red-700 text-red-300"
            : status === "flip"
            ? "bg-yellow-900/60 border-yellow-700 text-yellow-300"
            : "bg-gray-800 border-gray-700 text-gray-500";

        return (
          <div key={i} className="flex items-center">
            <div className={`px-2 py-1 rounded border text-xs font-medium whitespace-nowrap ${colors}`}>
              {node.icon} {node.name}
            </div>
            {i < GATE_NODES.length - 1 && (
              <div className="w-4 h-px bg-gray-700 mx-0.5" />
            )}
          </div>
        );
      })}
    </div>
  );
}