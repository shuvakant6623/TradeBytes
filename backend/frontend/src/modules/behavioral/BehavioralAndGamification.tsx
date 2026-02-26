import { useQuery } from '@tanstack/react-query';
import { behavioralApi, gamificationApi } from '../../shared/api';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, Tooltip, Cell,
} from 'recharts';

// ─── BEHAVIORAL ────────────────────────────────────────────────────────────

function BiasCard({ label, active, score }: { label: string; active: boolean; score: number }) {
  return (
    <div style={{
      padding: '14px 18px', borderRadius: 10,
      background: active ? '#FFF0F0' : '#F0FFF4',
      border: `1px solid ${active ? '#E74C3C' : '#27AE60'}`,
    }}>
      <div style={{ fontWeight: 700, color: active ? '#E74C3C' : '#27AE60', marginBottom: 4 }}>
        {active ? '⚠' : '✓'} {label}
      </div>
      <div style={{ fontSize: 13, color: '#666' }}>
        Score: <strong>{score.toFixed(3)}</strong>
      </div>
    </div>
  );
}

export function BehavioralDashboard({ userId }: { userId: string }) {
  const { data: profile } = useQuery({
    queryKey: ['behavioral', 'profile', userId],
    queryFn: () => behavioralApi.getProfile(userId),
  });

  const { data: biases } = useQuery({
    queryKey: ['behavioral', 'biases', userId],
    queryFn: () => behavioralApi.getBiases(userId),
  });

  if (!profile) return <div style={{ padding: 24 }}>Loading behavioral profile...</div>;

  const radarData = [
    { metric: 'Risk Score', value: (profile.risk_score || 0) / 100 * 100 },
    { metric: 'Profit Factor', value: Math.min((profile.profit_factor || 0) / 3 * 100, 100) },
    { metric: 'W/L Asymmetry', value: Math.min((profile.win_loss_asymmetry || 0) / 2 * 100, 100) },
    { metric: 'Diversification', value: (profile.diversification_beh || 0) * 100 },
    { metric: 'Recovery Speed', value: (profile.loss_recovery_speed || 0) * 100 },
  ];

  return (
    <div style={{ padding: 24, fontFamily: 'Arial, sans-serif' }}>
      <h2 style={{ color: '#1E3A5F', marginBottom: 8 }}>Behavioral Intelligence Engine</h2>
      <div style={{ marginBottom: 24 }}>
        <span style={{ background: '#E8F4FD', color: '#2E86AB', padding: '6px 16px', borderRadius: 20, fontWeight: 700 }}>
          🎭 {profile.archetype || 'Unknown Archetype'}
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 24 }}>
        <div style={{ background: '#F5F7FA', borderRadius: 12, padding: 20 }}>
          <h3 style={{ marginBottom: 12, color: '#1E3A5F' }}>Behavioural Radar</h3>
          <ResponsiveContainer width="100%" height={250}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11 }} />
              <Radar dataKey="value" stroke="#2E86AB" fill="#2E86AB" fillOpacity={0.3} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <h3 style={{ color: '#1E3A5F', margin: 0 }}>Bias Detection</h3>
          {biases && (
            <>
              <BiasCard
                label="Disposition Effect"
                active={biases.bias_flags?.disposition_effect}
                score={biases.scores?.disposition_effect || 0}
              />
              <BiasCard
                label="Overconfidence"
                active={biases.bias_flags?.overconfidence}
                score={biases.scores?.overconfidence || 0}
              />
              <BiasCard
                label="Loss Aversion"
                active={biases.bias_flags?.loss_aversion}
                score={biases.scores?.loss_aversion_ratio || 1}
              />
            </>
          )}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
        {[
          { label: 'Profit Factor', value: profile.profit_factor?.toFixed(2) },
          { label: 'Overtrading Z', value: profile.overtrading_z?.toFixed(2) },
          { label: 'Trade Duration P50 (min)', value: profile.trade_duration_p50?.toFixed(0) },
          { label: 'Risk Score', value: profile.risk_score?.toFixed(1) },
        ].map(({ label, value }) => (
          <div key={label} style={{ background: '#F5F7FA', borderRadius: 10, padding: '14px 16px', border: '1px solid #E0E8F0' }}>
            <div style={{ fontSize: 12, color: '#888', marginBottom: 4 }}>{label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: '#1E3A5F' }}>{value ?? '—'}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── GAMIFICATION ──────────────────────────────────────────────────────────

function XPProgressBar({ current, levelStart, levelEnd, level }: {
  current: number; levelStart: number; levelEnd: number; level: number;
}) {
  const pct = Math.min(100, Math.max(0, (current - levelStart) / (levelEnd - levelStart) * 100));
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
        <span style={{ fontWeight: 700, fontSize: 18, color: '#1E3A5F' }}>Level {level}</span>
        <span style={{ fontSize: 13, color: '#888' }}>{current.toLocaleString()} / {levelEnd.toLocaleString()} XP</span>
      </div>
      <div style={{ height: 12, background: '#E0E8F0', borderRadius: 6, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: 'linear-gradient(90deg, #2E86AB, #27AE60)', borderRadius: 6, transition: 'width 0.5s' }} />
      </div>
    </div>
  );
}

export function GamificationDashboard({ userId }: { userId: string }) {
  const { data: profile } = useQuery({
    queryKey: ['gamification', 'profile', userId],
    queryFn: () => gamificationApi.getProfile(userId),
    refetchInterval: 30_000,
  });

  const { data: leaderboard } = useQuery({
    queryKey: ['gamification', 'leaderboard'],
    queryFn: () => gamificationApi.getLeaderboard(),
    refetchInterval: 60_000,
  });

  const { data: xpHistory } = useQuery({
    queryKey: ['gamification', 'xpHistory', userId],
    queryFn: () => gamificationApi.getXpHistory(userId, 20),
  });

  if (!profile) return <div style={{ padding: 24 }}>Loading gamification data...</div>;

  const badges: any[] = profile.badges || [];

  return (
    <div style={{ padding: 24, fontFamily: 'Arial, sans-serif' }}>
      <h2 style={{ color: '#1E3A5F', marginBottom: 24 }}>Gamification Engine</h2>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 24 }}>
        <div style={{ background: '#F5F7FA', borderRadius: 12, padding: 20 }}>
          <XPProgressBar
            current={profile.total_xp || 0}
            levelStart={profile.current_level_xp || 0}
            levelEnd={profile.next_level_xp || 500}
            level={profile.level || 1}
          />
          <div style={{ display: 'flex', gap: 24, marginTop: 20 }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 28, fontWeight: 700, color: '#F39C12' }}>🔥 {profile.streak_days || 0}</div>
              <div style={{ fontSize: 12, color: '#888' }}>Day Streak</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 28, fontWeight: 700, color: '#27AE60' }}>{(profile.total_xp || 0).toLocaleString()}</div>
              <div style={{ fontSize: 12, color: '#888' }}>Total XP</div>
            </div>
          </div>
        </div>

        <div style={{ background: '#F5F7FA', borderRadius: 12, padding: 20 }}>
          <h3 style={{ marginBottom: 12, color: '#1E3A5F' }}>Badges ({badges.length})</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {badges.length === 0 && <span style={{ color: '#888', fontSize: 13 }}>No badges yet — start trading!</span>}
            {badges.map((b: any, i: number) => (
              <div key={i} style={{
                padding: '6px 12px', borderRadius: 20,
                background: '#E8F4FD', color: '#2E86AB', fontSize: 13, fontWeight: 600,
              }} title={new Date(b.earned_at).toLocaleDateString()}>
                {b.name}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
        <div style={{ background: '#F5F7FA', borderRadius: 12, padding: 20 }}>
          <h3 style={{ marginBottom: 12, color: '#1E3A5F' }}>Leaderboard (Top 10)</h3>
          {leaderboard?.leaderboard?.slice(0, 10).map((entry: any, i: number) => (
            <div key={i} style={{
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              padding: '8px 0', borderBottom: '1px solid #E0E8F0',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{ fontWeight: 700, color: i < 3 ? '#F39C12' : '#888', width: 24 }}>
                  {i + 1}
                </span>
                <span style={{ fontWeight: 600, color: '#1E3A5F' }}>{entry.username}</span>
                <span style={{ fontSize: 11, color: '#888' }}>Lv.{entry.level}</span>
              </div>
              <span style={{ fontWeight: 700, color: '#2E86AB' }}>{entry.score?.toFixed(2)}</span>
            </div>
          ))}
        </div>

        <div style={{ background: '#F5F7FA', borderRadius: 12, padding: 20 }}>
          <h3 style={{ marginBottom: 12, color: '#1E3A5F' }}>Recent XP Events</h3>
          {xpHistory?.xp_history?.map((tx: any, i: number) => (
            <div key={i} style={{
              display: 'flex', justifyContent: 'space-between',
              padding: '6px 0', borderBottom: '1px solid #E0E8F0', fontSize: 13,
            }}>
              <span style={{ color: '#444' }}>{tx.action_type.replace(/_/g, ' ')}</span>
              <span style={{ fontWeight: 700, color: '#27AE60' }}>+{tx.final_xp} XP</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
