import { useEffect, useState, useCallback } from "react";
import { Activity, TrendingUp, TrendingDown, RefreshCw, ExternalLink, Building2, BarChart3, Newspaper } from "lucide-react";

// ── Types ────────────────────────────────────────────────────────────────────

interface Index { name: string; ltp: number; open: number; high: number; low: number; prev_close: number; change: number; change_pct: number; }
interface Stock { symbol: string; ltp: number; prev_close: number; change: number; change_pct: number; high: number; low: number; }
interface FiiDii { buy: number; sell: number; net: number; date: string; }
interface NewsItem { title: string; source: string; published: string; url: string; }
interface StockDetail { fundamentals: Record<string, any>; ratings: Record<string, number> | null; price_history: { date: string; close: number }[]; }

// ── Formatters ───────────────────────────────────────────────────────────────

const fmt = (n: number) => `₹${n.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const fmtCr = (n: number) => `₹${(n / 10_000_000).toFixed(0)} Cr`;
const fmtLakh = (n: number) => `₹${(n / 100_000).toFixed(1)}L`;
const fmtMcap = (n: number) => n >= 1e12 ? `₹${(n / 1e12).toFixed(2)}T` : n >= 1e9 ? `₹${(n / 1e9).toFixed(0)}B` : fmtCr(n);
const pctClass = (n: number) => n > 0 ? "text-emerald-400" : n < 0 ? "text-red-400" : "text-gray-400";
const pctClassLight = (n: number) => n > 0 ? "text-emerald-500" : n < 0 ? "text-red-500" : "text-muted-foreground";
const arrow = (n: number) => n > 0 ? "▲" : n < 0 ? "▼" : "—";

export function Market() {
  const [indices, setIndices] = useState<Index[]>([]);
  const [gainers, setGainers] = useState<Stock[]>([]);
  const [losers, setLosers] = useState<Stock[]>([]);
  const [allStocks, setAllStocks] = useState<Stock[]>([]);
  const [fii, setFii] = useState<FiiDii | null>(null);
  const [dii, setDii] = useState<FiiDii | null>(null);
  const [news, setNews] = useState<NewsItem[]>([]);
  const [selectedStock, setSelectedStock] = useState<string | null>(null);
  const [stockDetail, setStockDetail] = useState<StockDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<string>("");

  const fetchAll = useCallback(async () => {
    try {
      const [idxRes, movRes, fiiRes, newsRes] = await Promise.all([
        fetch("/api/market/indices").then(r => r.json()).catch(() => ({ indices: [] })),
        fetch("/api/market/movers").then(r => r.json()).catch(() => ({ gainers: [], losers: [], all: [] })),
        fetch("/api/market/fiidii").then(r => r.json()).catch(() => ({})),
        fetch("/api/market/news").then(r => r.json()).catch(() => ({ articles: [] })),
      ]);
      setIndices(idxRes.indices || []);
      setGainers(movRes.gainers || []);
      setLosers(movRes.losers || []);
      setAllStocks(movRes.all || []);
      if (fiiRes.fii) setFii(fiiRes.fii);
      if (fiiRes.dii) setDii(fiiRes.dii);
      setNews(newsRes.articles || []);
      setLastUpdate(new Date().toLocaleTimeString("en-IN"));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchAll(); const t = setInterval(fetchAll, 30_000); return () => clearInterval(t); }, [fetchAll]);

  const fetchStockDetail = async (symbol: string) => {
    setSelectedStock(symbol);
    setDetailLoading(true);
    try {
      const res = await fetch(`/api/market/stock/${symbol}`);
      const data = await res.json();
      if (data.status === "ok") setStockDetail(data);
    } finally {
      setDetailLoading(false);
    }
  };

  if (loading) {
    return <div className="flex h-[80vh] items-center justify-center text-muted-foreground"><Activity className="h-6 w-6 animate-pulse mr-2" /> Loading market data...</div>;
  }

  return (
    <div className="flex flex-col h-full">
      {/* ── Index Ticker Bar ─────────────────────────────────────────── */}
      <div className="bg-gray-900 text-white px-4 py-2 flex items-center gap-6 overflow-x-auto shrink-0">
        {indices.map(idx => (
          <div key={idx.name} className="flex items-center gap-2 whitespace-nowrap">
            <span className="text-xs font-medium text-gray-400">{idx.name}</span>
            <span className="font-mono font-bold text-sm">{idx.ltp.toLocaleString("en-IN", { maximumFractionDigits: 2 })}</span>
            <span className={`text-xs font-mono ${pctClass(idx.change_pct)}`}>
              {arrow(idx.change_pct)} {Math.abs(idx.change).toFixed(2)} ({idx.change_pct > 0 ? "+" : ""}{idx.change_pct.toFixed(2)}%)
            </span>
          </div>
        ))}
        <div className="ml-auto flex items-center gap-2 text-xs text-gray-500">
          <span>Updated: {lastUpdate}</span>
          <button onClick={fetchAll} className="p-1 hover:text-white"><RefreshCw className="h-3 w-3" /></button>
        </div>
      </div>

      {/* ── Main Content ─────────────────────────────────────────────── */}
      <div className="flex-1 overflow-auto p-4">
        <div className="max-w-7xl mx-auto space-y-4">

          {/* Top row: Movers + FII/DII + Stock Detail */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

            {/* Gainers */}
            <div className="border rounded-lg overflow-hidden">
              <div className="bg-emerald-500/10 px-4 py-2 flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-emerald-500" />
                <span className="font-medium text-sm text-emerald-500">Top Gainers</span>
              </div>
              <div className="divide-y">
                {gainers.map(s => (
                  <button key={s.symbol} onClick={() => fetchStockDetail(s.symbol)}
                    className={`w-full px-4 py-2.5 flex items-center justify-between hover:bg-muted/50 transition-colors text-left ${selectedStock === s.symbol ? "bg-muted" : ""}`}>
                    <span className="font-medium text-sm">{s.symbol}</span>
                    <div className="text-right">
                      <div className="text-sm font-mono">{fmt(s.ltp)}</div>
                      <div className="text-xs font-mono text-emerald-500">+{s.change_pct.toFixed(2)}%</div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Losers */}
            <div className="border rounded-lg overflow-hidden">
              <div className="bg-red-500/10 px-4 py-2 flex items-center gap-2">
                <TrendingDown className="h-4 w-4 text-red-500" />
                <span className="font-medium text-sm text-red-500">Top Losers</span>
              </div>
              <div className="divide-y">
                {losers.map(s => (
                  <button key={s.symbol} onClick={() => fetchStockDetail(s.symbol)}
                    className={`w-full px-4 py-2.5 flex items-center justify-between hover:bg-muted/50 transition-colors text-left ${selectedStock === s.symbol ? "bg-muted" : ""}`}>
                    <span className="font-medium text-sm">{s.symbol}</span>
                    <div className="text-right">
                      <div className="text-sm font-mono">{fmt(s.ltp)}</div>
                      <div className="text-xs font-mono text-red-500">{s.change_pct.toFixed(2)}%</div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* FII/DII Activity */}
            <div className="border rounded-lg overflow-hidden">
              <div className="bg-blue-500/10 px-4 py-2 flex items-center gap-2">
                <Building2 className="h-4 w-4 text-blue-500" />
                <span className="font-medium text-sm text-blue-500">FII/DII Activity</span>
                {fii?.date && <span className="text-xs text-muted-foreground ml-auto">{fii.date}</span>}
              </div>
              <div className="p-4 space-y-4">
                {fii && (
                  <div>
                    <div className="flex justify-between text-xs text-muted-foreground mb-1"><span>FII/FPI</span><span className={pctClassLight(fii.net)}>Net: {fmtCr(fii.net)}</span></div>
                    <div className="flex gap-1 h-5">
                      <div className="bg-emerald-500/20 rounded" style={{ width: `${(fii.buy / (fii.buy + fii.sell)) * 100}%` }}>
                        <span className="text-[10px] px-1 text-emerald-600">Buy {fmtCr(fii.buy)}</span>
                      </div>
                      <div className="bg-red-500/20 rounded flex-1">
                        <span className="text-[10px] px-1 text-red-600">Sell {fmtCr(fii.sell)}</span>
                      </div>
                    </div>
                  </div>
                )}
                {dii && (
                  <div>
                    <div className="flex justify-between text-xs text-muted-foreground mb-1"><span>DII</span><span className={pctClassLight(dii.net)}>Net: {fmtCr(dii.net)}</span></div>
                    <div className="flex gap-1 h-5">
                      <div className="bg-emerald-500/20 rounded" style={{ width: `${(dii.buy / (dii.buy + dii.sell)) * 100}%` }}>
                        <span className="text-[10px] px-1 text-emerald-600">Buy {fmtCr(dii.buy)}</span>
                      </div>
                      <div className="bg-red-500/20 rounded flex-1">
                        <span className="text-[10px] px-1 text-red-600">Sell {fmtCr(dii.sell)}</span>
                      </div>
                    </div>
                  </div>
                )}
                {!fii && !dii && <div className="text-sm text-muted-foreground text-center py-4">FII/DII data unavailable</div>}
              </div>
            </div>
          </div>

          {/* Stock Detail Panel (shown when a stock is clicked) */}
          {selectedStock && (
            <div className="border rounded-lg overflow-hidden">
              <div className="bg-muted/50 px-4 py-2 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <BarChart3 className="h-4 w-4 text-primary" />
                  <span className="font-medium">{selectedStock} — Fundamentals & Ratings</span>
                </div>
                <button onClick={() => { setSelectedStock(null); setStockDetail(null); }} className="text-xs text-muted-foreground hover:text-foreground">✕ Close</button>
              </div>
              {detailLoading ? (
                <div className="p-6 text-center text-muted-foreground">Loading fundamentals...</div>
              ) : stockDetail ? (
                <div className="p-4 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
                  <MetricCard label="Market Cap" value={fmtMcap(stockDetail.fundamentals.market_cap || 0)} />
                  <MetricCard label="P/E (TTM)" value={stockDetail.fundamentals.pe_trailing?.toFixed(1) ?? "—"} />
                  <MetricCard label="P/E (Fwd)" value={stockDetail.fundamentals.pe_forward?.toFixed(1) ?? "—"} />
                  <MetricCard label="ROE" value={stockDetail.fundamentals.roe ? `${(stockDetail.fundamentals.roe * 100).toFixed(1)}%` : "—"} />
                  <MetricCard label="EPS" value={stockDetail.fundamentals.eps ? `₹${stockDetail.fundamentals.eps.toFixed(2)}` : "—"} />
                  <MetricCard label="Book Value" value={stockDetail.fundamentals.book_value ? `₹${stockDetail.fundamentals.book_value.toFixed(0)}` : "—"} />
                  <MetricCard label="Div Yield" value={stockDetail.fundamentals.dividend_yield ? `${stockDetail.fundamentals.dividend_yield.toFixed(2)}%` : "—"} />
                  <MetricCard label="D/E Ratio" value={stockDetail.fundamentals.debt_to_equity?.toFixed(1) ?? "—"} />
                  <MetricCard label="52W High" value={stockDetail.fundamentals.fifty_two_week_high ? fmt(stockDetail.fundamentals.fifty_two_week_high) : "—"} />
                  <MetricCard label="52W Low" value={stockDetail.fundamentals.fifty_two_week_low ? fmt(stockDetail.fundamentals.fifty_two_week_low) : "—"} />
                  <MetricCard label="Target Price" value={stockDetail.fundamentals.target_price ? fmt(stockDetail.fundamentals.target_price) : "—"} />
                  <MetricCard label="Recommendation"
                    value={stockDetail.fundamentals.recommendation?.replace("_", " ").toUpperCase() || "—"}
                    valueClass={stockDetail.fundamentals.recommendation?.includes("buy") ? "text-emerald-500" : stockDetail.fundamentals.recommendation?.includes("sell") ? "text-red-500" : ""} />
                  {stockDetail.ratings && (
                    <>
                      <div className="col-span-2 md:col-span-4 lg:col-span-6 border-t pt-3 mt-1">
                        <div className="text-xs font-medium text-muted-foreground mb-2">Analyst Ratings ({stockDetail.fundamentals.analyst_count} analysts)</div>
                        <div className="flex gap-1 h-6 rounded overflow-hidden">
                          {[
                            { key: "strong_buy", label: "Strong Buy", color: "bg-emerald-600" },
                            { key: "buy", label: "Buy", color: "bg-emerald-400" },
                            { key: "hold", label: "Hold", color: "bg-yellow-400" },
                            { key: "sell", label: "Sell", color: "bg-red-400" },
                            { key: "strong_sell", label: "Strong Sell", color: "bg-red-600" },
                          ].map(({ key, label, color }) => {
                            const count = stockDetail.ratings?.[key] || 0;
                            const total = Object.values(stockDetail.ratings || {}).reduce((a, b) => a + b, 0);
                            const pct = total > 0 ? (count / total) * 100 : 0;
                            return pct > 0 ? (
                              <div key={key} className={`${color} flex items-center justify-center`} style={{ width: `${pct}%` }}
                                title={`${label}: ${count}`}>
                                <span className="text-[10px] text-white font-medium">{count}</span>
                              </div>
                            ) : null;
                          })}
                        </div>
                        <div className="flex justify-between text-[10px] text-muted-foreground mt-1">
                          <span>Strong Buy</span><span>Buy</span><span>Hold</span><span>Sell</span><span>Strong Sell</span>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              ) : null}
            </div>
          )}

          {/* All Stocks Table */}
          <div className="border rounded-lg overflow-hidden">
            <div className="bg-muted/50 px-4 py-2 font-medium text-sm flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              NIFTY 50 Watchlist ({allStocks.length} stocks)
            </div>
            <div className="overflow-x-auto max-h-64">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-card">
                  <tr className="border-b">
                    <th className="text-left px-4 py-2 font-medium">Symbol</th>
                    <th className="text-right px-4 py-2 font-medium">LTP</th>
                    <th className="text-right px-4 py-2 font-medium">Change</th>
                    <th className="text-right px-4 py-2 font-medium">% Change</th>
                    <th className="text-right px-4 py-2 font-medium">High</th>
                    <th className="text-right px-4 py-2 font-medium">Low</th>
                  </tr>
                </thead>
                <tbody>
                  {allStocks.map(s => (
                    <tr key={s.symbol} className="border-b hover:bg-muted/30 cursor-pointer transition-colors" onClick={() => fetchStockDetail(s.symbol)}>
                      <td className="px-4 py-1.5 font-medium">{s.symbol}</td>
                      <td className="px-4 py-1.5 text-right font-mono">{fmt(s.ltp)}</td>
                      <td className={`px-4 py-1.5 text-right font-mono ${pctClassLight(s.change)}`}>{s.change > 0 ? "+" : ""}{s.change.toFixed(2)}</td>
                      <td className={`px-4 py-1.5 text-right font-mono font-medium ${pctClassLight(s.change_pct)}`}>{s.change_pct > 0 ? "+" : ""}{s.change_pct.toFixed(2)}%</td>
                      <td className="px-4 py-1.5 text-right font-mono text-muted-foreground">{fmt(s.high)}</td>
                      <td className="px-4 py-1.5 text-right font-mono text-muted-foreground">{fmt(s.low)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* ── News Terminal ──────────────────────────────────────────── */}
          <div className="border rounded-lg overflow-hidden bg-gray-950">
            <div className="bg-gray-900 px-4 py-2 flex items-center gap-2 border-b border-gray-800">
              <Newspaper className="h-4 w-4 text-emerald-400" />
              <span className="font-mono text-sm text-emerald-400">LIVE NEWS FEED</span>
              <span className="text-xs text-gray-600 ml-auto font-mono">{news.length} articles</span>
            </div>
            <div className="max-h-72 overflow-y-auto divide-y divide-gray-800">
              {news.length === 0 ? (
                <div className="px-4 py-8 text-center text-gray-600 font-mono text-sm">Loading news feed...</div>
              ) : (
                news.map((item, i) => (
                  <a key={i} href={item.url} target="_blank" rel="noopener noreferrer"
                    className="block px-4 py-2.5 hover:bg-gray-900/50 transition-colors group">
                    <div className="flex items-start gap-3">
                      <span className="text-emerald-500 font-mono text-xs mt-0.5 shrink-0">
                        {item.published ? new Date(item.published).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" }) : ""}
                      </span>
                      <div className="flex-1 min-w-0">
                        <p className="text-gray-200 text-sm font-mono leading-relaxed group-hover:text-emerald-300 transition-colors truncate">
                          {item.title}
                        </p>
                        <span className="text-xs text-gray-600 font-mono">{item.source}</span>
                      </div>
                      <ExternalLink className="h-3 w-3 text-gray-700 group-hover:text-emerald-500 shrink-0 mt-1" />
                    </div>
                  </a>
                ))
              )}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}

function MetricCard({ label, value, valueClass = "" }: { label: string; value: string; valueClass?: string }) {
  return (
    <div className="border rounded p-2.5">
      <div className="text-[10px] text-muted-foreground uppercase tracking-wider">{label}</div>
      <div className={`text-sm font-bold mt-0.5 ${valueClass}`}>{value}</div>
    </div>
  );
}
