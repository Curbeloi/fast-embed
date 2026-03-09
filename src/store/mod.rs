pub mod keys;
pub mod logs;

use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use std::path::Path;

pub type DbPool = Pool<SqliteConnectionManager>;

pub fn init_pool(db_path: &str) -> anyhow::Result<DbPool> {
    // Ensure parent directory exists
    if let Some(parent) = Path::new(db_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let manager = SqliteConnectionManager::file(db_path);
    let pool = Pool::builder().max_size(10).build(manager)?;

    // Run migrations
    run_migrations(&pool)?;

    Ok(pool)
}

fn run_migrations(pool: &DbPool) -> anyhow::Result<()> {
    let conn = pool.get()?;

    let keys_sql = include_str!("../../migrations/001_create_keys.sql");
    conn.execute_batch(keys_sql)?;

    let logs_sql = include_str!("../../migrations/002_create_logs.sql");
    conn.execute_batch(logs_sql)?;

    Ok(())
}
