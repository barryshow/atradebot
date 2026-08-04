#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Initialize the SQLite database for ATradeBot.
Run once before first use or after schema changes.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.engine.data.schema import init_db, SCHEMA_VERSION

def main():
    print(f"ATradeBot DB Init v{SCHEMA_VERSION}")
    conn = init_db()
    cur = conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
    version = cur.fetchone()[0]
    print(f"Schema version: {version}")

    # Verify tables
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    print(f"Tables: {', '.join(t[0] for t in tables)}")

    conn.close()
    print("Database initialized successfully.")

if __name__ == "__main__":
    main()
