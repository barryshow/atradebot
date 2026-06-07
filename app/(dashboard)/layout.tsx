import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "ATradeBot - AI量化战车",
  description: "AI+ML双轨制超短线量化交易系统",
};

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col min-h-screen bg-gray-950 text-gray-100">
      <header className="border-b border-gray-800 px-6 py-3">
        <div className="flex items-center gap-3">
          <span className="text-xl font-bold text-white">ATradeBot</span>
          <span className="text-xs text-gray-500 bg-gray-800 px-2 py-0.5 rounded">
            AI+ML 量化战车
          </span>
        </div>
      </header>
      <main className="flex-1 p-6">{children}</main>
    </div>
  );
}
