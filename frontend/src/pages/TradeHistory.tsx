import { useEffect, useState } from "react";
import { ArrowLeftRight, RefreshCw } from "lucide-react";

interface Order {
  id: string;
  symbol: string;
  exchange: string;
  side: string;
  order_type: string;
  qty: number;
  fill_price: number;
  slippage: number;
  fees_total: number;
  fees_breakdown: string | Record<string, number>;
  status: string;
  filled_qty: number;
  created_at: string;
  filled_at: string;
}

interface TradesData {
  orders: Order[];
  total_count: number;
}

const fmt = (n: number) => `₹${n.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

export function TradeHistory() {
  const [data, setData] = useState<TradesData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      const res = await fetch("/api/trades");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load trades");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, []);

  if (loading) {
    return <div className="flex h-[60vh] items-center justify-center text-muted-foreground">Loading trades…</div>;
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] gap-4">
        <p className="text-red-500">{error}</p>
        <button onClick={fetchData} className="px-4 py-2 rounded-md bg-primary text-primary-foreground text-sm">Retry</button>
      </div>
    );
  }

  const orders = data?.orders || [];
  const filledOrders = orders.filter(o => o.status === "FILLED");
  const totalFees = filledOrders.reduce((s, o) => s + (o.fees_total || 0), 0);
  const avgSlippage = filledOrders.length > 0
    ? filledOrders.reduce((s, o) => s + (o.slippage || 0), 0) / filledOrders.length
    : 0;

  return (
    <div className="flex flex-col gap-6 p-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <ArrowLeftRight className="h-6 w-6 text-primary" />
          <h1 className="text-2xl font-bold">Trade History</h1>
        </div>
        <button onClick={fetchData} className="flex items-center gap-2 px-3 py-1.5 rounded-md border text-sm text-muted-foreground hover:bg-muted transition-colors">
          <RefreshCw className="h-3.5 w-3.5" />
          Refresh
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4">
        <div className="border rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Total Trades</div>
          <div className="text-xl font-bold">{filledOrders.length}</div>
        </div>
        <div className="border rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Total Fees</div>
          <div className="text-xl font-bold">{fmt(totalFees)}</div>
        </div>
        <div className="border rounded-lg p-4">
          <div className="text-sm text-muted-foreground">Avg Slippage</div>
          <div className="text-xl font-bold">{fmt(avgSlippage)}</div>
        </div>
      </div>

      {/* Table */}
      <div className="border rounded-lg overflow-hidden">
        {orders.length === 0 ? (
          <div className="px-4 py-12 text-center text-muted-foreground">
            <ArrowLeftRight className="h-10 w-10 mx-auto mb-3 opacity-50" />
            <p>No trades yet</p>
            <p className="text-sm mt-1">Run <code className="bg-muted px-2 py-0.5 rounded">python run_paper_trading.py</code> to execute paper trades</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/30">
                  <th className="text-left px-4 py-2 font-medium">Time</th>
                  <th className="text-left px-4 py-2 font-medium">Symbol</th>
                  <th className="text-left px-4 py-2 font-medium">Side</th>
                  <th className="text-left px-4 py-2 font-medium">Type</th>
                  <th className="text-right px-4 py-2 font-medium">Qty</th>
                  <th className="text-right px-4 py-2 font-medium">Fill Price</th>
                  <th className="text-right px-4 py-2 font-medium">Slippage</th>
                  <th className="text-right px-4 py-2 font-medium">Fees</th>
                  <th className="text-left px-4 py-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {orders.map((o) => (
                  <tr key={o.id} className="border-b hover:bg-muted/20 transition-colors">
                    <td className="px-4 py-2.5 text-muted-foreground whitespace-nowrap">{(o.filled_at || o.created_at)?.slice(0, 16)}</td>
                    <td className="px-4 py-2.5 font-medium">{o.symbol}</td>
                    <td className="px-4 py-2.5">
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${o.side === "BUY" ? "bg-emerald-500/10 text-emerald-500" : "bg-red-500/10 text-red-500"}`}>
                        {o.side}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 text-muted-foreground">{o.order_type}</td>
                    <td className="px-4 py-2.5 text-right">{o.filled_qty || o.qty}</td>
                    <td className="px-4 py-2.5 text-right">{o.fill_price ? fmt(o.fill_price) : "—"}</td>
                    <td className="px-4 py-2.5 text-right text-muted-foreground">{o.slippage ? fmt(o.slippage) : "—"}</td>
                    <td className="px-4 py-2.5 text-right">{o.fees_total ? fmt(o.fees_total) : "—"}</td>
                    <td className="px-4 py-2.5">
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                        o.status === "FILLED" ? "bg-emerald-500/10 text-emerald-500" :
                        o.status === "REJECTED" ? "bg-red-500/10 text-red-500" :
                        "bg-yellow-500/10 text-yellow-500"
                      }`}>
                        {o.status}
                      </span>
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
