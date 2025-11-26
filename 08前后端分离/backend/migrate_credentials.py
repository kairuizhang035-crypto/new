#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import shutil
import stat

OLD_DIR = '/home/zkr/因果发现3/07分离'
SECURE_DIR = '/home/zkr/因果发现3/secure_credentials'
OLD_USERS_FILE = os.path.join(OLD_DIR, 'users.json')
OLD_LOG_FILE = os.path.join(OLD_DIR, 'users.log')
NEW_USERS_FILE = os.path.join(SECURE_DIR, 'users.json')
NEW_LOG_FILE = os.path.join(SECURE_DIR, 'users.log')

def ensure_secure_dir():
  os.makedirs(SECURE_DIR, exist_ok=True)
  try:
    os.chmod(SECURE_DIR, 0o700)
  except Exception:
    pass

def migrate():
  ensure_secure_dir()
  # users.json
  if os.path.exists(OLD_USERS_FILE):
    try:
      with open(OLD_USERS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    except Exception:
      data = {}
    try:
      with open(NEW_USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data or {}, f, ensure_ascii=False, indent=2)
      os.chmod(NEW_USERS_FILE, 0o600)
    except Exception:
      pass
  # users.log
  if os.path.exists(OLD_LOG_FILE):
    try:
      shutil.copy2(OLD_LOG_FILE, NEW_LOG_FILE)
      os.chmod(NEW_LOG_FILE, 0o600)
    except Exception:
      pass

if __name__ == '__main__':
  migrate()
  print('Credentials migrated to', SECURE_DIR)
