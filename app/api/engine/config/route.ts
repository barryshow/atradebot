import { NextRequest, NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    // 交易基础
    symbols: (process.env.TRADE_SYMBOLS || "BTCUSDT,ETHUSDT,SOLUSDT").split(","),
    holdMinutes: Number(process.env.HOLD_MINUTES) || 15,
    betMin: Number(process.env.BET_MIN) || 3,

    // L0: 防接刀
    antiKnifeCci: Number(process.env.ANTI_KNIFE_CCI) || 100,
    antiKnifeBodyRatio: Number(process.env.ANTI_KNIFE_BODY_RATIO) || 0.6,

    // L1: 硬性概率门槛
    hardProbThreshold: Number(process.env.HARD_PROB_THRESHOLD) || 0.62,

    // L2: 极值翻转
    bbExtremeHigh: Number(process.env.BB_EXTREME_HIGH) || 0.72,
    bbExtremeLow: Number(process.env.BB_EXTREME_LOW) || 0.28,
    reversalProb: Number(process.env.REVERSAL_PROB) || 0.55,

    // L3: 共振分
    confluenceMin: Number(process.env.CONFLUENCE_MIN) || 0.65,

    // L4: 双重冷却
    rejectCooldownSec: Number(process.env.REJECT_COOLDOWN_SEC) || 60,
    settlementCooldownSec: Number(process.env.SETTLEMENT_COOLDOWN_SEC) || 60,

    // L5: 加仓（已移除 — 二元期权每单独立，不做加仓）
    // addPositionMinRoi removed
  });
}

export async function PUT(req: NextRequest) {
  const body = await req.json();
  return NextResponse.json({ ok: true, msg: "Config update received", config: body });
}
