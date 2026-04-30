#!/usr/bin/env python3
"""
Lightweight GitHub webhook receiver.
Listens on port 9000 (loopback). Nginx proxies POST /deploy/webhook here.

On every push to main:
  git pull → docker compose build → docker compose up -d

Systemd unit: /etc/systemd/system/vzebra-webhook.service
"""
import os
import hmac
import hashlib
import json
import subprocess
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer

APP_DIR  = os.environ.get('APP_DIR',  '/home/ubuntu/vzebra')
SECRET   = os.environ.get('WEBHOOK_SECRET', '')  # set in systemd EnvironmentFile
BRANCH   = os.environ.get('DEPLOY_BRANCH', 'main')
PORT     = int(os.environ.get('WEBHOOK_PORT', '9000'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('webhook')


def _verify(body: bytes, sig_header: str) -> bool:
    if not SECRET:
        log.warning('WEBHOOK_SECRET not set — skipping signature check')
        return True
    expected = 'sha256=' + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig_header or '')


def _deploy():
    log.info('Deploy triggered — pulling and rebuilding…')
    cmds = [
        ['git', '-C', APP_DIR, 'pull', 'origin', BRANCH],
        ['docker', 'compose', '-f', f'{APP_DIR}/docker-compose.yml',
         'build', 'vzebra'],
        ['docker', 'compose', '-f', f'{APP_DIR}/docker-compose.yml',
         'up', '-d', 'vzebra'],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True)
        log.info(' '.join(cmd))
        if result.stdout: log.info(result.stdout.strip())
        if result.returncode != 0:
            log.error(result.stderr.strip())
            return False
    log.info('Deploy complete.')
    return True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.info(fmt % args)

    def do_POST(self):
        if self.path != '/deploy/webhook':
            self.send_response(404); self.end_headers(); return

        length = int(self.headers.get('Content-Length', 0))
        body   = self.rfile.read(length)
        sig    = self.headers.get('X-Hub-Signature-256', '')

        if not _verify(body, sig):
            log.warning('Bad signature — rejected')
            self.send_response(403); self.end_headers(); return

        try:
            payload = json.loads(body)
        except Exception:
            self.send_response(400); self.end_headers(); return

        ref = payload.get('ref', '')
        if ref != f'refs/heads/{BRANCH}':
            log.info('Push to %s — ignoring (not %s)', ref, BRANCH)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'ignored')
            return

        self.send_response(202)
        self.end_headers()
        self.wfile.write(b'deploying')
        _deploy()


if __name__ == '__main__':
    log.info('Webhook server listening on 127.0.0.1:%d', PORT)
    HTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
