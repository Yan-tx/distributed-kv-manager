#!/usr/bin/env python3
"""Clear KV metadata in etcd and stored KV files on disk.

Usage:
  python scripts/clear_kv_store.py [--yes] [--dry-run] [--session-prefix PREFIX] [--etcd-only] [--storage-only]

By default the script reads `config.json` in the repo root to discover
`kv_transfer_config.etcd_endpoints` and `kv_transfer_config.storage_dir`.

Options:
  --yes                Actually perform deletions. Without this the script runs in dry-run mode.
  --dry-run            Explicit dry run (default if --yes omitted).
  --session-prefix STR Only delete metadata/files whose key/filename contains this prefix (e.g. 'kv_session_0000').
  --etcd-only          Only touch etcd (don't remove files on disk).
  --storage-only       Only remove files on disk (don't touch etcd).
  --ssd                Also clean ssd_cache_dir if present in config.

This is a convenience script intended for development/testing. Use with care.
"""
import argparse
import json
import os
import sys
import fnmatch


def load_config(repo_root=None):
    if repo_root is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cfg_path = os.path.join(repo_root, 'config.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f'config.json not found at {cfg_path}')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    return cfg.get('kv_transfer_config', {})


def clear_storage(storage_dir, session_prefix=None, dry_run=True, also_ssd=None):
    storage_dir = os.path.expanduser(storage_dir)
    if not os.path.exists(storage_dir):
        print(f'[storage] storage dir not found: {storage_dir}')
        return 0
    removed = 0
    print(f'[storage] scanning {storage_dir} (dry_run={dry_run})')
    for root, dirs, files in os.walk(storage_dir):
        for fn in files:
            # target files typically look like: kv_<session>_layer_<id>_*.pt
            if fn.startswith('kv_') and fn.endswith('.pt'):
                if session_prefix and session_prefix not in fn:
                    continue
                path = os.path.join(root, fn)
                if dry_run:
                    print('[storage] would remove', path)
                else:
                    try:
                        os.remove(path)
                        print('[storage] removed', path)
                        removed += 1
                    except Exception as e:
                        print('[storage] failed to remove', path, e)
    if also_ssd:
        ssd_dir = os.path.expanduser(also_ssd)
        if os.path.exists(ssd_dir):
            for root, dirs, files in os.walk(ssd_dir):
                for fn in files:
                    if fn.endswith('.pt') and (not session_prefix or session_prefix in fn):
                        path = os.path.join(root, fn)
                        if dry_run:
                            print('[ssd] would remove', path)
                        else:
                            try:
                                os.remove(path)
                                print('[ssd] removed', path)
                                removed += 1
                            except Exception as e:
                                print('[ssd] failed to remove', path, e)
    return removed


def clear_etcd(endpoints, session_prefix=None, dry_run=True):
    # endpoints: list like ['127.0.0.1:2379']
    try:
        import etcd3
    except Exception as e:
        print('[etcd] etcd3 package not available:', e)
        print("Install with: pip install etcd3")
        return 0

    total_deleted = 0
    for ep in endpoints:
        host, sep, port = ep.partition(':')
        port = int(port) if sep else 2379
        print(f'[etcd] connecting to {host}:{port} (dry_run={dry_run})')
        try:
            client = etcd3.client(host=host, port=port)
        except Exception as e:
            print('[etcd] failed to connect to', ep, e)
            continue

        prefix = '/kvmeta'
        print(f'[etcd] listing keys with prefix {prefix}')
        try:
            it = client.get_prefix(prefix)
            for value, meta in it:
                # meta is an etcd3.metadata.KeyValue
                key = meta.key.decode() if isinstance(meta.key, bytes) else meta.key
                if session_prefix and session_prefix not in key:
                    continue
                if dry_run:
                    print('[etcd] would delete', key)
                else:
                    ok = client.delete(key)
                    print('[etcd] deleted', key, 'ok=', ok)
                    total_deleted += 1
        except Exception as e:
            print('[etcd] error listing/deleting keys at', ep, e)
    return total_deleted


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--yes', action='store_true', help='Actually perform deletions.')
    p.add_argument('--dry-run', action='store_true', help='Dry run (default).')
    p.add_argument('--session-prefix', default=None, help='Only delete keys/files containing this prefix')
    p.add_argument('--etcd-only', action='store_true')
    p.add_argument('--storage-only', action='store_true')
    p.add_argument('--ssd', action='store_true', help='Also clean ssd cache dir from config if present')
    args = p.parse_args()

    dry = not args.yes or args.dry_run

    cfg = load_config()
    etcd_eps = cfg.get('etcd_endpoints', [])
    storage_dir = cfg.get('storage_dir') or cfg.get('local_dir')
    ssd_dir = cfg.get('ssd_cache_dir') if args.ssd else None

    if not args.storage_only:
        if not etcd_eps:
            print('[etcd] no endpoints found in config, skipping etcd')
        else:
            n = clear_etcd(etcd_eps, session_prefix=args.session_prefix, dry_run=dry)
            print('[etcd] total deleted:', n)

    if not args.etcd_only:
        if not storage_dir:
            print('[storage] storage_dir not found in config, skipping storage cleanup')
        else:
            n = clear_storage(storage_dir, session_prefix=args.session_prefix, dry_run=dry, also_ssd=ssd_dir)
            print('[storage] total removed:', n)

    if dry:
        print('\nDRY RUN completed. To actually delete, re-run with --yes')


if __name__ == '__main__':
    main()
