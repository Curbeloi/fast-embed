use crate::store::DbPool;
use anyhow::Result;
use chrono::Utc;
use rusqlite::params;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, serde::Serialize)]
pub struct KeyInfo {
    pub id: String,
    pub tenant_id: String,
    pub name: String,
    pub active: bool,
    pub rate_limit_override: Option<u32>,
    pub created_at: String,
    pub revoked_at: Option<String>,
    pub last_used_at: Option<String>,
}

pub fn hash_key(key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    hex::encode(hasher.finalize())
}

pub fn generate_key(prefix: &str) -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let random_bytes: Vec<u8> = (0..16).map(|_| rng.gen()).collect();
    format!("{}_{}", prefix, hex::encode(random_bytes))
}

pub fn validate_key(key: &str, pool: &DbPool) -> Result<Option<KeyInfo>> {
    let conn = pool.get()?;
    let key_hash = hash_key(key);

    let mut stmt = conn.prepare(
        "SELECT id, tenant_id, name, active, rate_limit_override, created_at, revoked_at, last_used_at
         FROM api_keys WHERE key_hash = ?1 AND active = 1",
    )?;

    let result = stmt.query_row(params![key_hash], |row| {
        Ok(KeyInfo {
            id: row.get(0)?,
            tenant_id: row.get(1)?,
            name: row.get(2)?,
            active: row.get::<_, i32>(3)? == 1,
            rate_limit_override: row.get::<_, Option<i32>>(4)?.map(|v| v as u32),
            created_at: row.get(5)?,
            revoked_at: row.get(6)?,
            last_used_at: row.get(7)?,
        })
    });

    match result {
        Ok(info) => {
            // Update last_used_at
            let now = Utc::now().to_rfc3339();
            let _ = conn.execute(
                "UPDATE api_keys SET last_used_at = ?1 WHERE key_hash = ?2",
                params![now, key_hash],
            );
            Ok(Some(info))
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

pub fn create_key(
    tenant_id: &str,
    name: &str,
    rate_limit_override: Option<u32>,
    pool: &DbPool,
) -> Result<(String, KeyInfo)> {
    let conn = pool.get()?;
    let id = uuid::Uuid::now_v7().to_string();
    let raw_key = generate_key("cg_prod");
    let key_hash = hash_key(&raw_key);
    let now = Utc::now().to_rfc3339();

    conn.execute(
        "INSERT INTO api_keys (id, key_hash, name, tenant_id, active, rate_limit_override, created_at)
         VALUES (?1, ?2, ?3, ?4, 1, ?5, ?6)",
        params![id, key_hash, name, tenant_id, rate_limit_override.map(|v| v as i32), now],
    )?;

    let info = KeyInfo {
        id,
        tenant_id: tenant_id.to_string(),
        name: name.to_string(),
        active: true,
        rate_limit_override,
        created_at: now,
        revoked_at: None,
        last_used_at: None,
    };

    Ok((raw_key, info))
}

pub fn create_admin_key(pool: &DbPool) -> Result<(String, KeyInfo)> {
    let conn = pool.get()?;
    let id = uuid::Uuid::now_v7().to_string();
    let raw_key = generate_key("cg_admin");
    let key_hash = hash_key(&raw_key);
    let now = Utc::now().to_rfc3339();

    conn.execute(
        "INSERT INTO api_keys (id, key_hash, name, tenant_id, active, created_at)
         VALUES (?1, ?2, 'Admin Key', '__admin__', 1, ?3)",
        params![id, key_hash, now],
    )?;

    let info = KeyInfo {
        id,
        tenant_id: "__admin__".to_string(),
        name: "Admin Key".to_string(),
        active: true,
        rate_limit_override: None,
        created_at: now,
        revoked_at: None,
        last_used_at: None,
    };

    Ok((raw_key, info))
}

pub fn revoke_key(key_id: &str, pool: &DbPool) -> Result<bool> {
    let conn = pool.get()?;
    let now = Utc::now().to_rfc3339();
    let rows = conn.execute(
        "UPDATE api_keys SET active = 0, revoked_at = ?1 WHERE id = ?2 AND active = 1",
        params![now, key_id],
    )?;
    Ok(rows > 0)
}

pub fn list_keys(pool: &DbPool) -> Result<Vec<KeyInfo>> {
    let conn = pool.get()?;
    let mut stmt = conn.prepare(
        "SELECT id, tenant_id, name, active, rate_limit_override, created_at, revoked_at, last_used_at
         FROM api_keys ORDER BY created_at DESC",
    )?;

    let keys = stmt
        .query_map([], |row| {
            Ok(KeyInfo {
                id: row.get(0)?,
                tenant_id: row.get(1)?,
                name: row.get(2)?,
                active: row.get::<_, i32>(3)? == 1,
                rate_limit_override: row.get::<_, Option<i32>>(4)?.map(|v| v as u32),
                created_at: row.get(5)?,
                revoked_at: row.get(6)?,
                last_used_at: row.get(7)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(keys)
}

pub fn is_admin_key(key: &str, pool: &DbPool) -> Result<bool> {
    let conn = pool.get()?;
    let key_hash = hash_key(key);
    let count: i32 = conn.query_row(
        "SELECT COUNT(*) FROM api_keys WHERE key_hash = ?1 AND active = 1 AND tenant_id = '__admin__'",
        params![key_hash],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}
