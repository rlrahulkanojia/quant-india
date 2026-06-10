import { useEffect, useState } from "react";
import { Wallet, TrendingUp, TrendingDown, DollarSign, RefreshCw } from "lucide-react";

interface Position {
  id: number;
  symbol: string;
  exchange: string;
  qty: number;
  avg_price: number;
  current_price: number;
  unrealized_pnl: number;
  side: string;
  opened_at: string;
}

interface PortfolioData {
  cash: number;
  positions: Position[];
  unrealized_pnl: number;
  realized_pnl: number;
  total_value: number;
  total_fees_paid: number;
  initialized: boolean;
}

const fmt = (n: number) => `₹${n.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const pnlColor = (n: number) => n > 0 ? "text-emerald-500" : n < 0 ? "text-red-500" : "text-muted-foreground";

export function Portfolio() {
  const [data, setData] = useState<PortfolioData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      const res = await fetch("/api/portfolio");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load portfolio");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30_000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex h-[60vh] items-center justify-center text-muted-foreground">
        Loading portfolio…
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] gap-4">
        <p className="text-red-500">{error}</p>
        <button onClick={fetchData} className="px-4 py-2 rounded-md bg-primary text-primary-foreground text-sm">
          Retry
        </button>
      </div>
    );
  }

  if (!data || !data.initialized) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] gap-4 text-muted-foreground">
        <Wallet className="h-12 w-12" />
        <p className="text-lg">No paper portfolio yet</p>
        <p className="text-sm">Run <code className="bg-muted px-2 py-1 rounded">python run_paper_trading.py</code> to create your first trades</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6 p-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Wallet className="h-6 w-6 text-primary" />
          <h1 className="text-2xl font-bold">Paper Portfolio</h1>
        </div>
        <button
          onClick={fetchData}
          className="flex items-center gap-2 px-3 py-1.5 rounded-md border text-sm text-muted-foreground hover:bg-muted transition-colors"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          Refresh
        </button>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card label="Cash Balance" value={fmt(data.cash)} icon={DollarSign} />
        <Card label="Unrealized P&L" value={fmt(data.unrealized_pnl)} icon={TrendingUp} valueClass={pnlColor(data.unrealized_pnl)} />
        <Card label="Realized P&L" value={fmt(data.realized_pnl)} icon={TrendingDown} valueClass={pnlColor(data.realized_pnl)} />
        <Card label="Total Value" value={fmt(data.total_value)} icon={Wallet} />
      </div>

      {/* Fees */}
      <div className="text-sm text-muted-foreground">
        Total fees paid: {fmt(data.total_fees_paid)}
      </div>

      {/* Positions Table */}
      <div className="border rounded-lg overflow-hidden">
        <div className="bg-muted/50 px-4 py-3 font-medium text-sm">
          Open Positions ({data.positions.length})
        </div>
        {data.positions.length === 0 ? (
          <div className="px-4 py-8 text-center text-muted-foreground text-sm">
            No open positions
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/30">
                  <th className="text-left px-4 py-2 font-medium">Symbol</th>
                  <th className="text-left px-4 py-2 font-medium">Side</th>
                  <th className="text-right px-4 py-2 font-medium">Qty</th>
                  <th className="text-right px-4 py-2 font-medium">Avg Price</th>
                  <th className="text-right px-4 py-2 font-medium">Current</th>
                  <th className="text-right px-4 py-2 font-medium">P&L</th>
                  <th className="text-left px-4 py-2 font-medium">Opened</th>
                </tr>
              </thead>
              <tbody>
                {data.positions.map((p) => {
                  const pnl = p.side === "LONG"
                    ? (p.current_price - p.avg_price) * p.qty
                    : (p.avg_price - p.current_price) * p.qty;
                  return (
                    <tr key={p.id} className="border-b hover:bg-muted/20 transition-colors">
                      <td className="px-4 py-2.5 font-medium">{p.symbol}</td>
                      <td className="px-4 py-2.5">
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${p.side === "LONG" ? "bg-emerald-500/10 text-emerald-500" : "bg-red-500/10 text-red-500"}`}>
                          {p.side}
                        </span>
                      </td>
                      <td className="px-4 py-2.5 text-right">{p.qty}</td>
                      <td className="px-4 py-2.5 text-right">{fmt(p.avg_price)}</td>
                      <td className="px-4 py-2.5 text-right">{p.current_price ? fmt(p.current_price) : "—"}</td>
                      <td className={`px-4 py-2.5 text-right font-medium ${pnlColor(pnl)}`}>
                        {fmt(pnl)}
                      </td>
                      <td className="px-4 py-2.5 text-muted-foreground">{p.opened_at?.slice(0, 16)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function Card({ label, value, icon: Icon, valueClass = "" }: { label: string; value: string; icon: any; valueClass?: string }) {
  return (
    <div className="border rounded-lg p-4">
      <div className="flex items-center gap-2 text-sm text-muted-foreground mb-1">
        <Icon className="h-4 w-4" />
        {label}
      </div>
      <div className={`text-xl font-bold ${valueClass}`}>{value}</div>
    </div>
  );
}
