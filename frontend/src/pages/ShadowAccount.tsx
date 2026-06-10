import { useEffect, useState } from "react";
import { Eye, RefreshCw } from "lucide-react";

interface ShadowFill {
  paper_order_id: string;
  paper_fill_price: number;
  market_ltp: number;
  market_bid: number;
  market_ask: number;
  divergence_pct: number;
  qty: number;
  captured_at: string;
}

interface ShadowData {
  fills: ShadowFill[];
  avg_divergence_pct: number;
  max_divergence_pct: number;
  fill_count: number;
  paper_total: number;
  market_total: number;
  divergence_cost: number;
}

const fmt = (n: number) => `₹${n.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const pct = (n: number) => `${n >= 0 ? "+" : ""}${n.toFixed(4)}%`;

export function ShadowAccount() {
  const [data, setData] = useState<ShadowData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [days, setDays] = useState(7);

  const fetchData = async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/shadow?days=${days}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load shadow data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, [days]);

  if (loading) {
    return <div className="flex h-[60vh] items-center justify-center text-muted-foreground">Loading shadow account…</div>;
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] gap-4">
        <p className="text-red-500">{error}</p>
        <button onClick={fetchData} className="px-4 py-2 rounded-md bg-primary text-primary-foreground text-sm">Retry</button>
      </div>
    );
  }

  const d = data!;

  return (
    <div className="flex flex-col gap-6 p-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Eye className="h-6 w-6 text-primary" />
          <h1 className="text-2xl font-bold">Shadow Account</h1>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex gap-1.5">
            {[7, 14, 30].map((n) => (
              <button
                key={n}
                onClick={() => setDays(n)}
                className={`px-3 py-1.5 rounded-md text-sm transition-colors ${
                  days === n ? "bg-primary text-primary-foreground" : "border text-muted-foreground hover:bg-muted"
                }`}
              >
                {n}d
              </button>
            ))}
          </div>
          <button onClick={fetchData} className="flex items-center gap-2 px-3 py-1.5 rounded-md border text-sm text-muted-foreground hover:bg-muted transition-colors">
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Explanation */}
      <div className="text-sm text-muted-foreground border-l-2 border-primary/30 pl-3">
        Shadow account compares your paper fill prices against what the real market price was at that exact moment.
        Divergence shows how realistic your paper trading simulation is.
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="border rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Fills Tracked</div>
          <div className="text-xl font-bold">{d.fill_count}</div>
        </div>
        <div className="border rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Avg Divergence</div>
          <div className={`text-xl font-bold ${d.avg_divergence_pct > 0.1 ? "text-yellow-500" : "text-emerald-500"}`}>
            {pct(d.avg_divergence_pct)}
          </div>
        </div>
        <div className="border rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Max Divergence</div>
          <div className={`text-xl font-bold ${d.max_divergence_pct > 0.5 ? "text-red-500" : "text-yellow-500"}`}>
            {pct(d.max_divergence_pct)}
          </div>
        </div>
        <div className="border rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Divergence Cost</div>
          <div className={`text-xl font-bold ${d.divergence_cost > 0 ? "text-red-500" : "text-emerald-500"}`}>
            {fmt(d.divergence_cost)}
          </div>
        </div>
      </div>

      {/* Paper vs Market Totals */}
      {d.fill_count > 0 && (
        <div className="grid grid-cols-2 gap-4">
          <div className="border rounded-lg p-4">
            <div className="text-sm text-muted-foreground">Paper Total Value</div>
            <div className="text-lg font-bold">{fmt(d.paper_total)}</div>
          </div>
          <div className="border rounded-lg p-4">
            <div className="text-sm text-muted-foreground">Market Total Value (what live would be)</div>
            <div className="text-lg font-bold">{fmt(d.market_total)}</div>
          </div>
        </div>
      )}

      {/* Fills Table */}
      <div className="border rounded-lg overflow-hidden">
        <div className="bg-muted/50 px-4 py-3 font-medium text-sm">
          Fill Comparisons ({d.fills.length})
        </div>
        {d.fills.length === 0 ? (
          <div className="px-4 py-12 text-center text-muted-foreground">
            <Eye className="h-10 w-10 mx-auto mb-3 opacity-50" />
            <p>No shadow fills in the last {days} days</p>
            <p className="text-sm mt-1">Execute paper trades to start tracking divergence</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/30">
                  <th className="text-left px-4 py-2 font-medium">Time</th>
                  <th className="text-left px-4 py-2 font-medium">Order</th>
                  <th className="text-right px-4 py-2 font-medium">Paper Price</th>
                  <th className="text-right px-4 py-2 font-medium">Market LTP</th>
                  <th className="text-right px-4 py-2 font-medium">Bid</th>
                  <th className="text-right px-4 py-2 font-medium">Ask</th>
                  <th className="text-right px-4 py-2 font-medium">Divergence</th>
                </tr>
              </thead>
              <tbody>
                {d.fills.map((f, i) => (
                  <tr key={i} className="border-b hover:bg-muted/20 transition-colors">
                    <td className="px-4 py-2.5 text-muted-foreground whitespace-nowrap">{f.captured_at?.slice(0, 19)}</td>
                    <td className="px-4 py-2.5 font-mono text-xs">{f.paper_order_id?.slice(0, 25)}…</td>
                    <td className="px-4 py-2.5 text-right">{fmt(f.paper_fill_price)}</td>
                    <td className="px-4 py-2.5 text-right">{fmt(f.market_ltp)}</td>
                    <td className="px-4 py-2.5 text-right text-muted-foreground">{fmt(f.market_bid)}</td>
                    <td className="px-4 py-2.5 text-right text-muted-foreground">{fmt(f.market_ask)}</td>
                    <td className={`px-4 py-2.5 text-right font-medium ${Math.abs(f.divergence_pct) > 0.1 ? "text-yellow-500" : "text-emerald-500"}`}>
                      {pct(f.divergence_pct)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
