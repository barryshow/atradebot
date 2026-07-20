#!/usr/bin/env python3
"""Retrain Fast Entry models after price feature fix"""
import subprocess, sys
sys.exit(subprocess.run([sys.executable, "scripts/train_fast_entry.py", "--days", "60", "--symbols", "BTCUSDT,ETHUSDT,SOLUSDT"]).returncode)