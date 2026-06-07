import { NextRequest, NextResponse } from "next/server";
import { writeFileSync, existsSync, readFileSync } from "fs";
import { join } from "path";

interface HibtConfig {
  token: string;
  authorization: string;
  xAuthToken: string;
  bgetKey: string;
  bgetId: string;
  feishuWebhook: string;
}

export async function POST(req: NextRequest) {
  const body: HibtConfig = await req.json();

  // 更新 .env.local
  const envPath = join(process.cwd(), ".env.local");
  let content = "";
  if (existsSync(envPath)) {
    content = readFileSync(envPath, "utf-8");
  }

  const updates: Record<string, string> = {
    HIBT_TOKEN: body.token,
    HIBT_AUTHORIZATION: body.authorization,
    HIBT_X_AUTH_TOKEN: body.xAuthToken,
    HIBT_BGET_KEY: body.bgetKey,
    HIBT_BGET_ID: body.bgetId,
    FEISHU_WEBHOOK: body.feishuWebhook,
  };

  for (const [key, val] of Object.entries(updates)) {
    if (!val) continue;
    const regex = new RegExp(`^${key}=.*`, "m");
    if (regex.test(content)) {
      content = content.replace(regex, `${key}=${val}`);
    } else {
      content += `\n${key}=${val}`;
    }
  }

  writeFileSync(envPath, content, "utf-8");

  return NextResponse.json({ ok: true });
}