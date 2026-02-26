import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useState } from 'react';
import { RegimeDashboard } from './modules/regime/RegimeDashboard';
import { NewsDashboard } from './modules/news/NewsDashboard';
import { RiskDashboard } from './modules/risk/RiskDashboard';
import { BehavioralDashboard, GamificationDashboard } from './modules/behavioral/BehavioralAndGamification';

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 30_000, retry: 1 } },
});

const MODULES = [
  { id: 'regime',       label: '📊 Regime',      icon: '📊' },
  { id: 'news',         label: '📰 News',         icon: '📰' },
  { id: 'risk',         label: '⚡ Risk',          icon: '⚡' },
  { id: 'behavioral',   label: '🧠 Behavioral',   icon: '🧠' },
  { id: 'gamification', label: '🎮 Gamification', icon: '🎮' },
];

const DEMO_USER_ID = '00000000-0000-0000-0000-000000000001';

function Sidebar({ active, onSelect }: { active: string; onSelect: (id: string) => void }) {
  return (
    <div style={{
      width: 200, minHeight: '100vh', background: '#1E3A5F',
      padding: '24px 0', display: 'flex', flexDirection: 'column',
    }}>
      <div style={{ padding: '0 20px 32px', borderBottom: '1px solid #2E4E7A' }}>
        <div style={{ color: '#2E86AB', fontSize: 20, fontWeight: 900 }}>AIFIP</div>
        <div style={{ color: '#888', fontSize: 11, marginTop: 2 }}>Financial Intelligence</div>
      </div>
      <nav style={{ padding: '16px 0' }}>
        {MODULES.map(m => (
          <button
            key={m.id}
            onClick={() => onSelect(m.id)}
            style={{
              display: 'block', width: '100%', textAlign: 'left',
              padding: '12px 20px', border: 'none', cursor: 'pointer',
              background: active === m.id ? '#2E4E7A' : 'transparent',
              color: active === m.id ? '#fff' : '#AAC0D5',
              fontSize: 14, fontWeight: active === m.id ? 600 : 400,
              borderLeft: active === m.id ? '3px solid #2E86AB' : '3px solid transparent',
            }}
          >
            {m.label}
          </button>
        ))}
      </nav>
    </div>
  );
}

function MainContent({ moduleId }: { moduleId: string }) {
  switch (moduleId) {
    case 'regime':       return <RegimeDashboard assetId="SPY" />;
    case 'news':         return <NewsDashboard />;
    case 'risk':         return <RiskDashboard userId={DEMO_USER_ID} />;
    case 'behavioral':   return <BehavioralDashboard userId={DEMO_USER_ID} />;
    case 'gamification': return <GamificationDashboard userId={DEMO_USER_ID} />;
    default:             return <RegimeDashboard assetId="SPY" />;
  }
}

export default function App() {
  const [activeModule, setActiveModule] = useState('regime');

  return (
    <QueryClientProvider client={queryClient}>
      <div style={{ display: 'flex', fontFamily: 'Arial, sans-serif', minHeight: '100vh', background: '#F0F4F8' }}>
        <Sidebar active={activeModule} onSelect={setActiveModule} />
        <main style={{ flex: 1, overflowY: 'auto', maxHeight: '100vh' }}>
          <MainContent moduleId={activeModule} />
        </main>
      </div>
    </QueryClientProvider>
  );
}
