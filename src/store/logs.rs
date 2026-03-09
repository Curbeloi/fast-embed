use crate::store::DbPool;
use anyhow::Result;
use rusqlite::params;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct RequestLog {
    pub request_id: String,
    pub api_key_id: String,
    pub tenant_id: String,
    pub endpoint: String,
    pub model: String,
    pub tokens_in: u32,
    pub tokens_out: u32,
    pub duration_ms: u64,
    pub device: String,
    pub status: u16,
    pub error: Option<String>,
    pub timestamp: String,
}

pub fn insert(log: &RequestLog, pool: &DbPool) -> Result<()> {
    let conn = pool.get()?;
    conn.execute(
        "INSERT INTO request_logs (id, api_key_id, tenant_id, endpoint, model, tokens_in, tokens_out, duration_ms, device, status, error, timestamp)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![
            log.request_id,
            log.api_key_id,
            log.tenant_id,
            log.endpoint,
            log.model,
            log.tokens_in,
            log.tokens_out,
            log.duration_ms,
            log.device,
            log.status,
            log.error,
            log.timestamp,
        ],
    )?;
    Ok(())
}

#[derive(Debug, Serialize)]
pub struct TenantStats {
    pub tenant_id: String,
    pub requests: u64,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub avg_duration_ms: f64,
    pub errors: u64,
}

#[derive(Debug, Serialize)]
pub struct AggregateStats {
    pub total_requests: u64,
    pub total_tokens_in: u64,
    pub total_tokens_out: u64,
    pub total_errors: u64,
    pub avg_duration_ms: f64,
    pub by_tenant: Vec<TenantStats>,
}

pub fn get_stats(pool: &DbPool, from: &str, to: &str) -> Result<AggregateStats> {
    let conn = pool.get()?;

    // Totals
    let mut stmt = conn.prepare(
        "SELECT COUNT(*), COALESCE(SUM(tokens_in),0), COALESCE(SUM(tokens_out),0),
                COALESCE(AVG(duration_ms),0), COALESCE(SUM(CASE WHEN status >= 400 THEN 1 ELSE 0 END),0)
         FROM request_logs WHERE timestamp >= ?1 AND timestamp <= ?2",
    )?;

    let (total_requests, total_tokens_in, total_tokens_out, avg_duration_ms, total_errors): (
        u64,
        u64,
        u64,
        f64,
        u64,
    ) = stmt.query_row(params![from, to], |row| {
        Ok((
            row.get(0)?,
            row.get(1)?,
            row.get(2)?,
            row.get(3)?,
            row.get(4)?,
        ))
    })?;

    // By tenant
    let mut stmt = conn.prepare(
        "SELECT tenant_id, COUNT(*), COALESCE(SUM(tokens_in),0), COALESCE(SUM(tokens_out),0),
                COALESCE(AVG(duration_ms),0), COALESCE(SUM(CASE WHEN status >= 400 THEN 1 ELSE 0 END),0)
         FROM request_logs WHERE timestamp >= ?1 AND timestamp <= ?2
         GROUP BY tenant_id ORDER BY COUNT(*) DESC",
    )?;

    let by_tenant = stmt
        .query_map(params![from, to], |row| {
            Ok(TenantStats {
                tenant_id: row.get(0)?,
                requests: row.get(1)?,
                tokens_in: row.get(2)?,
                tokens_out: row.get(3)?,
                avg_duration_ms: row.get(4)?,
                errors: row.get(5)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(AggregateStats {
        total_requests,
        total_tokens_in,
        total_tokens_out,
        total_errors,
        avg_duration_ms,
        by_tenant,
    })
}

pub fn get_tenant_stats(pool: &DbPool, tenant_id: &str, from: &str, to: &str) -> Result<Option<TenantStats>> {
    let conn = pool.get()?;
    let mut stmt = conn.prepare(
        "SELECT tenant_id, COUNT(*), COALESCE(SUM(tokens_in),0), COALESCE(SUM(tokens_out),0),
                COALESCE(AVG(duration_ms),0), COALESCE(SUM(CASE WHEN status >= 400 THEN 1 ELSE 0 END),0)
         FROM request_logs WHERE tenant_id = ?1 AND timestamp >= ?2 AND timestamp <= ?3
         GROUP BY tenant_id",
    )?;

    let result = stmt.query_row(params![tenant_id, from, to], |row| {
        Ok(TenantStats {
            tenant_id: row.get(0)?,
            requests: row.get(1)?,
            tokens_in: row.get(2)?,
            tokens_out: row.get(3)?,
            avg_duration_ms: row.get(4)?,
            errors: row.get(5)?,
        })
    });

    match result {
        Ok(stats) => Ok(Some(stats)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}
