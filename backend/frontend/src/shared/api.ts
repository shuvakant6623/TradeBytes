import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
});

// ─── REGIME ────────────────────────────────────────────────────────────────
export const regimeApi = {
  getCurrent: (assetId = 'SPY') =>
    api.get(`/api/v1/regime/current?asset_id=${assetId}`).then(r => r.data),
  getHistory: (assetId = 'SPY', limit = 100) =>
    api.get(`/api/v1/regime/history?asset_id=${assetId}&limit=${limit}`).then(r => r.data),
  getTransitions: () =>
    api.get('/api/v1/regime/transitions').then(r => r.data),
  retrain: (assetId = 'SPY') =>
    api.post('/api/v1/regime/retrain', { asset_id: assetId }).then(r => r.data),
};

// ─── NEWS ──────────────────────────────────────────────────────────────────
export const newsApi = {
  getFeed: (ticker?: string, sentiment?: string, limit = 50) =>
    api.get('/api/v1/news/feed', { params: { ticker, sentiment, limit } }).then(r => r.data),
  getSentiment: (ticker: string, hours = 168) =>
    api.get(`/api/v1/news/sentiment/${ticker}?hours=${hours}`).then(r => r.data),
  getCorrelation: (ticker: string, lagBars = 1, windowDays = 30) =>
    api.get(`/api/v1/news/correlation/${ticker}?lag_bars=${lagBars}&window_days=${windowDays}`).then(r => r.data),
  semanticSearch: (query: string, ticker?: string, limit = 10) =>
    api.post('/api/v1/news/search/semantic', { query, ticker, limit }).then(r => r.data),
  getImpact: (ticker: string, days = 30) =>
    api.get(`/api/v1/news/impact/${ticker}?days=${days}`).then(r => r.data),
};

// ─── RISK ──────────────────────────────────────────────────────────────────
export const riskApi = {
  getSummary: (userId: string, windowDays = 252) =>
    api.get(`/api/v1/risk/summary/${userId}?window_days=${windowDays}`).then(r => r.data),
  getCorrelation: (userId: string) =>
    api.get(`/api/v1/risk/correlation/${userId}`).then(r => r.data),
  getDrawdown: (userId: string, days = 252) =>
    api.get(`/api/v1/risk/drawdown/${userId}?days=${days}`).then(r => r.data),
  getHealth: (userId: string) =>
    api.get(`/api/v1/risk/health/${userId}`).then(r => r.data),
  simulate: (userId: string, positions: Record<string, number>) =>
    api.post('/api/v1/risk/simulate', { user_id: userId, hypothetical_positions: positions }).then(r => r.data),
};

// ─── BEHAVIORAL ────────────────────────────────────────────────────────────
export const behavioralApi = {
  getProfile: (userId: string) =>
    api.get(`/api/v1/behavioral/profile/${userId}`).then(r => r.data),
  getBiases: (userId: string) =>
    api.get(`/api/v1/behavioral/biases/${userId}`).then(r => r.data),
  recalculate: (userId: string) =>
    api.post(`/api/v1/behavioral/recalculate?user_id=${userId}`).then(r => r.data),
};

// ─── GAMIFICATION ──────────────────────────────────────────────────────────
export const gamificationApi = {
  getProfile: (userId: string) =>
    api.get(`/api/v1/gamification/profile/${userId}`).then(r => r.data),
  getLeaderboard: (cohort?: string, limit = 50) =>
    api.get('/api/v1/gamification/leaderboard', { params: { cohort, limit } }).then(r => r.data),
  getXpHistory: (userId: string, limit = 50) =>
    api.get(`/api/v1/gamification/xp/history?user_id=${userId}&limit=${limit}`).then(r => r.data),
  getUnlocks: (userId: string) =>
    api.get(`/api/v1/gamification/unlocks?user_id=${userId}`).then(r => r.data),
};
