import { useQuery } from '@tanstack/react-query';
import { regimeApi } from '../../shared/api';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, CartesianGrid, Cell,
} from 'recharts';

const REGIME_COLORS: Record<string, string> = {
  trending: '#27AE60',
  mean_reverting: '#3498DB',
  high_volatility: '#E74C3C',
  low_volatility: '#95A5A6',
  news_driven: '#F39C12',
};

const REGIME_LABELS: Record<string, string> = {
  trending: 'Trending',
  mean_reverting: 'Mean-Reverting',
  high_volatility: 'High Volatility',
  low_volatility: 'Low Volatility',
  news_driven: 'News-Driven',
};

function RegimeBadge({ regime, confidence }: { regime: string; confidence: number }) {
  const color = REGIME_COLORS[regime] || '#888';
  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center', gap: 10,
      padding: '12px 24px', borderRadius: 12,
      backgroundColor: color + '22', border: `2px solid ${color}`,
    }}>
      <div style={{ width: 12, height: 12, borderRadius: '50%', backgroundColor: color }} />
      <span style={{ fontWeight: 700, fontSize: 20, color }}>{REGIME_LABELS[regime] || regime}</span>
      <span style={{ fontSize: 14, color: '#666' }}>{(confidence * 100).toFixed(0)}% confidence</span>
    </div>
  );
}

function ConfidenceBar({ confidence }: { confidence: Record<string, number> }) {
  const data = Object.entries(confidence).map(([regime, prob]) => ({
    regime: REGIME_LABELS[regime] || regime,
    probability: +(prob * 100).toFixed(1),
    color: REGIME_COLORS[regime] || '#888',
  })).sort((a, b) => b.probability - a.probability);

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} layout="vertical" margin={{ left: 20 }}>
        <XAxis type="number" domain={[0, 100]} tickFormatter={v => `${v}%`} />
        <YAxis type="category" dataKey="regime" width={120} />
        <Tooltip formatter={(v: number) => `${v.toFixed(1)}%`} />
        <Bar dataKey="probability" radius={[0, 4, 4, 0]}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function RegimeTimeline({ history }: { history: any[] }) {
  const data = [...history].reverse().slice(-60).map((h: any) => ({
    time: new Date(h.time).toLocaleDateString(),
    regime: h.regime,
    color: REGIME_COLORS[h.regime] || '#888',
  }));

  return (
    <div style={{ marginTop: 24 }}>
      <h3 style={{ marginBottom: 12, color: '#1E3A5F' }}>Regime Timeline (60 bars)</h3>
      <div style={{ display: 'flex', gap: 2, height: 40, alignItems: 'center' }}>
        {data.map((d, i) => (
          <div
            key={i}
            title={`${d.time}: ${REGIME_LABELS[d.regime] || d.regime}`}
            style={{
              flex: 1, height: '100%', backgroundColor: d.color,
              borderRadius: 2, opacity: 0.8, cursor: 'pointer',
            }}
          />
        ))}
      </div>
      <div style={{ display: 'flex', gap: 16, marginTop: 12, flexWrap: 'wrap' }}>
        {Object.entries(REGIME_COLORS).map(([regime, color]) => (
          <div key={regime} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 12, height: 12, backgroundColor: color, borderRadius: 2 }} />
            <span style={{ fontSize: 12, color: '#666' }}>{REGIME_LABELS[regime]}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function RegimeDashboard({ assetId = 'SPY' }: { assetId?: string }) {
  const { data: current, isLoading: loadingCurrent } = useQuery({
    queryKey: ['regime', 'current', assetId],
    queryFn: () => regimeApi.getCurrent(assetId),
    refetchInterval: 60_000,
  });

  const { data: history } = useQuery({
    queryKey: ['regime', 'history', assetId],
    queryFn: () => regimeApi.getHistory(assetId, 100),
    refetchInterval: 300_000,
  });

  const { data: transitions } = useQuery({
    queryKey: ['regime', 'transitions'],
    queryFn: () => regimeApi.getTransitions(),
    staleTime: 3_600_000,
  });

  if (loadingCurrent) return <div style={{ padding: 24 }}>Loading regime data...</div>;

  const topRegime = current?.regime;
  const topConfidence = topRegime ? current.confidence[topRegime] : 0;

  return (
    <div style={{ padding: 24, fontFamily: 'Arial, sans-serif' }}>
      <h2 style={{ color: '#1E3A5F', marginBottom: 8 }}>Market Regime Detection</h2>
      <p style={{ color: '#666', marginBottom: 24 }}>
        Asset: <strong>{assetId}</strong> · Model: {current?.model_version || 'unknown'}
      </p>

      {topRegime && (
        <div style={{ marginBottom: 24 }}>
          <RegimeBadge regime={topRegime} confidence={topConfidence} />
        </div>
      )}

      {current?.confidence && (
        <div style={{ background: '#F5F7FA', borderRadius: 12, padding: 20, marginBottom: 24 }}>
          <h3 style={{ marginBottom: 12, color: '#1E3A5F' }}>State Probabilities</h3>
          <ConfidenceBar confidence={current.confidence} />
        </div>
      )}

      {history?.history && (
        <RegimeTimeline history={history.history} />
      )}

      {transitions && (
        <div style={{ marginTop: 24, background: '#F5F7FA', borderRadius: 12, padding: 20 }}>
          <h3 style={{ marginBottom: 12, color: '#1E3A5F' }}>State Map</h3>
          <pre style={{ fontSize: 12, color: '#444', overflow: 'auto' }}>
            {JSON.stringify(transitions.state_map, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
