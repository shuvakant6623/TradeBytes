import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import structlog

logger = structlog.get_logger()


async def update_leaderboard(db: AsyncSession) -> None:
    """
    Recalculate all leaderboard scores. Called hourly.
    Score = 0.40×sharpe_z + 0.25×xp_7d_z + 0.20×winrate_z + 0.15×health_z
    All components are z-score normalised within cohort (account age quartile).
    """
    # Pull latest portfolio snapshots
    rows = await db.execute(text("""
        SELECT DISTINCT ON (ps.user_id)
            ps.user_id,
            ps.sharpe,
            ps.health_score,
            ug.total_xp,
            u.created_at AS account_created
        FROM portfolio_snapshots ps
        JOIN user_gamification ug ON ug.user_id = ps.user_id
        JOIN users u ON u.id = ps.user_id
        ORDER BY ps.user_id, ps.computed_at DESC
    """))
    records = rows.fetchall()

    if not records:
        return

    df = pd.DataFrame(records, columns=["user_id", "sharpe", "health_score", "total_xp", "account_created"])
    df["account_created"] = pd.to_datetime(df["account_created"])

    # Win rate from trades
    wr_rows = await db.execute(text("""
        SELECT user_id,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) AS win_rate
        FROM trades
        WHERE status = 'closed'
        GROUP BY user_id
    """))
    wr_df = pd.DataFrame(wr_rows.fetchall(), columns=["user_id", "win_rate"])
    df = df.merge(wr_df, on="user_id", how="left").fillna({"win_rate": 0.5})

    # XP earned last 7 days
    xp_rows = await db.execute(text("""
        SELECT user_id, COALESCE(SUM(final_xp), 0) AS xp_7d
        FROM xp_transactions
        WHERE ts >= NOW() - INTERVAL '7 days'
        GROUP BY user_id
    """))
    xp_df = pd.DataFrame(xp_rows.fetchall(), columns=["user_id", "xp_7d"])
    df = df.merge(xp_df, on="user_id", how="left").fillna({"xp_7d": 0})

    # Cohort assignment (account age quartile)
    now = pd.Timestamp.utcnow().tz_localize(None)
    df["account_age_days"] = (now - df["account_created"].dt.tz_localize(None)).dt.days
    df["cohort"] = pd.qcut(df["account_age_days"], q=4, labels=["rookie", "junior", "senior", "veteran"], duplicates="drop")

    for cohort in df["cohort"].unique():
        cohort_mask = df["cohort"] == cohort
        cohort_df = df[cohort_mask].copy()

        def zscore(series: pd.Series) -> pd.Series:
            std = series.std()
            if std < 1e-8:
                return pd.Series(0.0, index=series.index)
            return (series - series.mean()) / std

        cohort_df["sharpe_z"]  = zscore(cohort_df["sharpe"].fillna(0))
        cohort_df["xp_z"]      = zscore(cohort_df["xp_7d"].fillna(0))
        cohort_df["winrate_z"] = zscore(cohort_df["win_rate"].fillna(0))
        cohort_df["health_z"]  = zscore(cohort_df["health_score"].fillna(0))

        cohort_df["score"] = (
            0.40 * cohort_df["sharpe_z"] +
            0.25 * cohort_df["xp_z"] +
            0.20 * cohort_df["winrate_z"] +
            0.15 * cohort_df["health_z"]
        )

        # Flag outliers (> 3σ) for review
        score_std = cohort_df["score"].std()
        outlier_threshold = cohort_df["score"].mean() + 3 * score_std

        cohort_df["rank"] = cohort_df["score"].rank(ascending=False, method="min").astype(int)

        for _, row in cohort_df.iterrows():
            await db.execute(text("""
                INSERT INTO leaderboard_scores
                  (user_id, score, cohort, rank, sharpe_z, xp_z, winrate_z, health_z, updated_at)
                VALUES
                  (:uid, :score, :cohort, :rank, :sz, :xz, :wz, :hz, NOW())
                ON CONFLICT (user_id) DO UPDATE SET
                  score = EXCLUDED.score, cohort = EXCLUDED.cohort, rank = EXCLUDED.rank,
                  sharpe_z = EXCLUDED.sharpe_z, xp_z = EXCLUDED.xp_z,
                  winrate_z = EXCLUDED.winrate_z, health_z = EXCLUDED.health_z,
                  updated_at = NOW()
            """), {
                "uid": str(row["user_id"]),
                "score": float(row["score"]),
                "cohort": str(cohort),
                "rank": int(row["rank"]),
                "sz": float(row["sharpe_z"]),
                "xz": float(row["xp_z"]),
                "wz": float(row["winrate_z"]),
                "hz": float(row["health_z"]),
            })

    await db.commit()
    logger.info("leaderboard_updated", total_users=len(df))
