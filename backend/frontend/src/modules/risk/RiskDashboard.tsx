import { useQuery } from '@tanstack/react-query';
import { riskApi } from '../../shared/api';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, RadialBarChart, RadialBar,
} from 'recharts';

function HealthGauge({ score }: { score: number }) {
  const color = score >= 70 ? '#27AE60' : score >= 40 ? '#F39C12' : '#E74C3C';
  const data = [{ name: 'health', value: score, fill: color }];

  return (
    <div style={{ textAlign: 'center' }}>
      <ResponsiveContainer width={200} height={200}>
        <RadialBarChart
          cx="50%" cy="50%"
          innerRadius="60%" outerRadius="90%"
          data={data}
          startAngle={180} endAngle={0}
        >
          <RadialBar dataKey="value" cornerRadius={10} background={{ fill: '#eee' }} />
        </RadialBarChart>
      </ResponsiveContainer>
      <div style={{ marginTop: -60 }}>
        <div style={{ fontSize: 36, fontWeight: 700, color }}>{score.toFixed(0)}</div>
        <div style={{ fontSize: 14, color: '#666' }}>Portfolio Health</div>
      </div>
    </div>
  );
}

function MetricCard({ label, value, unit = '', color = '#1E3A5F' }: {
  label: string; value: number | null; unit?: string; color?: string;
}) {
  return (
    <div style={{
      background: '#F5F7FA', borderRadius: 12, padding: '16px 20px',
      border: '1px solid #E0E8F0',
    }}>
      <div style={{ fontSize: 13, color: '#888', marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 700, color }}>
        {value !== null && value !== undefined ? `${value.toFixed(3)}${unit}` : '—'}
      </div>
    </div>
  );
}

function DrawdownChart({ series }: { series: { date: string; drawdown: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <AreaChart data={series} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
        <XAxis dataKey="date" tick={{ fontSize: 11 }} tickCount={6} />
        <YAxis tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 11 }} />
        <Tooltip formatter={(v: number) => `${(v * 100).toFixed(2)}%`} />
        <Area
          type="monotone" dataKey="drawdown"
          stroke="#E74C3C" fill="#E74C3C" fillOpacity={0.3}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function CorrelationHeatmap({ matrix }: { matrix: Record<string, Record<string, number>> }) {
  const assets = Object.keys(matrix);
  if (assets.length === 0) return null;

  const colorScale = (v: number) => {
    const r = v > 0 ? Math.round(255 * v) : 0;
    const b = v < 0 ? Math.round(255 * -v) : 0;
    const g = Math.round(255 * (1 - Math.abs(v)) * 0.3);
    return `rgb(${r},${g},${b})`;
  };

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr>
            <th style={{ padding: '6px 10px' }} />
            {assets.map(a => (
              <th key={a} style={{ padding: '6px 10px', color: '#444', fontWeight: 600 }}>{a}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {assets.map(rowAsset => (
            <tr key={rowAsset}>
              <td style={{ padding: '6px 10px', fontWeight: 600, color: '#444' }}>{rowAsset}</td>
              {assets.map(colAsset => {
                const val = matrix[rowAsset]?.[colAsset] ?? 0;
                return (
                  <td
                    key={colAsset}
                    style={{
                      padding: '8px 12px', textAlign: 'center',
                      backgroundColor: colorScale(val),
                      color: Math.abs(val) > 0.5 ? '#fff' : '#333',
                      fontWeight: rowAsset === colAsset ? 700 : 400,
                    }}
                    title={`${rowAsset} vs ${colAsset}: ${val.toFixed(3)}`}
                  >
                    {val.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function RiskDashboard({ userId }: { userId: string }) {
  const { data: summary } = useQuery({
    queryKey: ['risk', 'summary', userId],
    queryFn: () => riskApi.getSummary(userId),
    refetchInterval: 300_000,
  });

  const { data: drawdownData } = useQuery({
    queryKey: ['risk', 'drawdown', userId],
    queryFn: () => riskApi.getDrawdown(userId),
  });

  const { data: corrData } = useQuery({
    queryKey: ['risk', 'correlation', userId],
    queryFn: () => riskApi.getCorrelation(userId),
  });

  if (!summary) return <div style={{ padding: 24 }}>Loading risk data...</div>;

  return (
    <div style={{ padding: 24, fontFamily: 'Arial, sans-serif' }}>
      <h2 style={{ color: '#1E3A5F', marginBottom: 24 }}>Portfolio Risk Engine</h2>

      <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start', marginBottom: 32 }}>
        {summary.health_score !== null && (
          <HealthGauge score={summary.health_score} />
        )}
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)',
          gap: 16, flex: 1,
        }}>
          <MetricCard label="Sharpe Ratio" value={summary.sharpe} />
          <MetricCard label="Annualised Volatility" value={summary.volatility} unit="σ" />
          <MetricCard label="Beta (vs SPY)" value={summary.beta} />
          <MetricCard label="Max Drawdown" value={summary.max_drawdown ? summary.max_drawdown * 100 : null} unit="%" color="#E74C3C" />
          <MetricCard label="Diversification Index" value={summary.diversification} />
          <MetricCard label="HHI (Concentration)" value={summary.hhi} />
        </div>
      </div>

      {summary.warnings?.length > 0 && (
        <div style={{ background: '#FFF3CD', border: '1px solid #F39C12', borderRadius: 8, padding: 16, marginBottom: 24 }}>
          {summary.warnings.map((w: any, i: number) => (
            <div key={i} style={{ color: '#856404' }}>⚠ {w.message}</div>
          ))}
        </div>
      )}

      {drawdownData?.drawdown_series && (
        <div style={{ background: '#F5F7FA', borderRadius: 12, padding: 20, marginBottom: 24 }}>
          <h3 style={{ marginBottom: 12, color: '#1E3A5F' }}>
            Drawdown — Max: {(drawdownData.max_drawdown * 100).toFixed(2)}%
          </h3>
          <DrawdownChart series={drawdownData.drawdown_series} />
        </div>
      )}

      {corrData?.correlation_matrix && Object.keys(corrData.correlation_matrix).length > 1 && (
        <div style={{ background: '#F5F7FA', borderRadius: 12, padding: 20 }}>
          <h3 style={{ marginBottom: 12, color: '#1E3A5F' }}>Correlation Matrix</h3>
          <CorrelationHeatmap matrix={corrData.correlation_matrix} />
        </div>
      )}
    </div>
  );
}
