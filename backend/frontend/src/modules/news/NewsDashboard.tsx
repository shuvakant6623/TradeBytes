import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { newsApi } from '../../shared/api';
import {
  ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';

const SENTIMENT_COLORS = {
  positive: '#27AE60',
  neutral: '#95A5A6',
  negative: '#E74C3C',
};

function SentimentBadge({ label }: { label: string }) {
  const color = SENTIMENT_COLORS[label as keyof typeof SENTIMENT_COLORS] || '#888';
  return (
    <span style={{
      padding: '2px 8px', borderRadius: 4, fontSize: 11,
      backgroundColor: color + '22', color, fontWeight: 600,
    }}>
      {label}
    </span>
  );
}

function NewsFeedItem({ article }: { article: any }) {
  return (
    <div style={{
      padding: 16, borderBottom: '1px solid #E8EEF5',
      display: 'flex', justifyContent: 'space-between', gap: 16,
    }}>
      <div style={{ flex: 1 }}>
        <a href={article.url} target="_blank" rel="noreferrer"
           style={{ color: '#1E3A5F', fontWeight: 600, textDecoration: 'none', fontSize: 14 }}>
          {article.headline}
        </a>
        <div style={{ marginTop: 6, display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
          <span style={{ fontSize: 12, color: '#888' }}>{article.source}</span>
          <span style={{ fontSize: 12, color: '#888' }}>
            {new Date(article.published_at).toLocaleString()}
          </span>
          {article.tickers?.filter(Boolean).map((t: string) => (
            <span key={t} style={{ fontSize: 11, background: '#E8F4FD', color: '#2E86AB', padding: '1px 6px', borderRadius: 3 }}>
              ${t}
            </span>
          ))}
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4 }}>
        <SentimentBadge label={article.sentiment_label} />
        <span style={{ fontSize: 13, fontWeight: 700, color: article.sentiment_score > 0 ? '#27AE60' : article.sentiment_score < 0 ? '#E74C3C' : '#888' }}>
          {article.sentiment_score > 0 ? '+' : ''}{(article.sentiment_score * 100).toFixed(0)}
        </span>
      </div>
    </div>
  );
}

function SentimentChart({ ticker }: { ticker: string }) {
  const { data } = useQuery({
    queryKey: ['news', 'sentiment', ticker],
    queryFn: () => newsApi.getSentiment(ticker, 168),
    enabled: !!ticker,
  });

  if (!data?.sentiment_series) return null;

  const chartData = data.sentiment_series.map((d: any) => ({
    time: new Date(d.bucket).toLocaleDateString(),
    avg_sentiment: +(d.avg_sentiment * 100).toFixed(1),
    articles: d.article_count,
  }));

  return (
    <div style={{ background: '#F5F7FA', borderRadius: 12, padding: 20, marginBottom: 24 }}>
      <h3 style={{ marginBottom: 12, color: '#1E3A5F' }}>7-Day Sentiment — ${ticker}</h3>
      <ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
          <XAxis dataKey="time" tick={{ fontSize: 11 }} />
          <YAxis yAxisId="left" tickFormatter={v => `${v}`} tick={{ fontSize: 11 }} />
          <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} />
          <Tooltip />
          <ReferenceLine y={0} yAxisId="left" stroke="#888" strokeDasharray="4 4" />
          <Bar yAxisId="right" dataKey="articles" fill="#B8D9F0" opacity={0.6} />
          <Line yAxisId="left" type="monotone" dataKey="avg_sentiment" stroke="#2E86AB" dot={false} strokeWidth={2} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function SemanticSearch() {
  const [query, setQuery] = useState('');
  const [submitted, setSubmitted] = useState('');

  const { data, isLoading } = useQuery({
    queryKey: ['news', 'semantic', submitted],
    queryFn: () => newsApi.semanticSearch(submitted),
    enabled: !!submitted,
  });

  return (
    <div style={{ marginBottom: 24 }}>
      <h3 style={{ marginBottom: 12, color: '#1E3A5F' }}>Semantic News Search</h3>
      <div style={{ display: 'flex', gap: 10 }}>
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="e.g. Fed rate decision impact on tech stocks..."
          style={{ flex: 1, padding: '10px 14px', borderRadius: 8, border: '1px solid #C8D8E8', fontSize: 14 }}
          onKeyDown={e => e.key === 'Enter' && setSubmitted(query)}
        />
        <button
          onClick={() => setSubmitted(query)}
          style={{ padding: '10px 20px', background: '#2E86AB', color: '#fff', border: 'none', borderRadius: 8, cursor: 'pointer' }}
        >
          Search
        </button>
      </div>
      {isLoading && <div style={{ marginTop: 12, color: '#888' }}>Searching...</div>}
      {data?.results?.map((r: any, i: number) => (
        <div key={i} style={{ marginTop: 8, padding: 12, background: '#F5F7FA', borderRadius: 8 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ fontWeight: 600, fontSize: 13, color: '#1E3A5F' }}>{r.headline}</span>
            <span style={{ fontSize: 12, color: '#888' }}>sim: {(r.similarity * 100).toFixed(0)}%</span>
          </div>
          <div style={{ marginTop: 4 }}>
            <SentimentBadge label={r.sentiment_label} />
            <span style={{ marginLeft: 8, fontSize: 12, color: '#888' }}>{new Date(r.published_at).toLocaleDateString()}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

export function NewsDashboard() {
  const [ticker, setTicker] = useState('AAPL');
  const [activeTicker, setActiveTicker] = useState('AAPL');
  const [sentimentFilter, setSentimentFilter] = useState<string>('');

  const { data: feed } = useQuery({
    queryKey: ['news', 'feed', activeTicker, sentimentFilter],
    queryFn: () => newsApi.getFeed(activeTicker, sentimentFilter || undefined, 50),
    refetchInterval: 60_000,
  });

  return (
    <div style={{ padding: 24, fontFamily: 'Arial, sans-serif' }}>
      <h2 style={{ color: '#1E3A5F', marginBottom: 24 }}>Financial News Intelligence</h2>

      <div style={{ display: 'flex', gap: 10, marginBottom: 24, alignItems: 'center' }}>
        <input
          value={ticker}
          onChange={e => setTicker(e.target.value.toUpperCase())}
          placeholder="Ticker (e.g. AAPL)"
          style={{ padding: '8px 12px', borderRadius: 8, border: '1px solid #C8D8E8', width: 120 }}
        />
        <button
          onClick={() => setActiveTicker(ticker)}
          style={{ padding: '8px 16px', background: '#1E3A5F', color: '#fff', border: 'none', borderRadius: 8, cursor: 'pointer' }}
        >
          Load
        </button>
        <select
          value={sentimentFilter}
          onChange={e => setSentimentFilter(e.target.value)}
          style={{ padding: '8px 12px', borderRadius: 8, border: '1px solid #C8D8E8' }}
        >
          <option value="">All Sentiment</option>
          <option value="positive">Positive</option>
          <option value="neutral">Neutral</option>
          <option value="negative">Negative</option>
        </select>
      </div>

      <SentimentChart ticker={activeTicker} />
      <SemanticSearch />

      <div style={{ background: '#fff', border: '1px solid #E0E8F0', borderRadius: 12 }}>
        <div style={{ padding: 16, borderBottom: '1px solid #E0E8F0', fontWeight: 600, color: '#1E3A5F' }}>
          News Feed — {feed?.count ?? '...'} articles
        </div>
        {feed?.articles?.map((a: any) => (
          <NewsFeedItem key={a.id} article={a} />
        ))}
      </div>
    </div>
  );
}
