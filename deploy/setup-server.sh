#!/bin/bash
# One-time setup for vzebra on the Lightsail server.
# Run as: bash setup-server.sh
set -e

REPO="https://github.com/psalmhim/vzebra.git"
APP_DIR="/home/ubuntu/vzebra"
DOMAIN="vzebra.xaiforbrain.com"
EMAIL="hangik.lee@gmail.com"
WEBHOOK_SECRET="${WEBHOOK_SECRET:-$(openssl rand -hex 32)}"

echo "=== [1/6] Clone or update repo ==="
if [ -d "$APP_DIR/.git" ]; then
    git -C "$APP_DIR" pull origin main
else
    git clone "$REPO" "$APP_DIR"
fi

echo "=== [2/6] Build Docker image (first build ~10 min for PyTorch) ==="
cd "$APP_DIR"
docker compose build

echo "=== [3/6] Start app container ==="
docker compose up -d
sleep 8
curl -sf http://localhost:5001/api/status && echo " OK" || echo " WARNING: server not responding yet"

echo "=== [4/6] Install nginx config ==="
sudo cp "$APP_DIR/deploy/nginx-vzebra.conf" \
        "/etc/nginx/sites-available/$DOMAIN"
sudo ln -sf "/etc/nginx/sites-available/$DOMAIN" \
            "/etc/nginx/sites-enabled/$DOMAIN"
sudo nginx -t && sudo systemctl reload nginx

echo "=== [5/6] Issue SSL certificate ==="
sudo certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$EMAIL"
sudo systemctl reload nginx

echo "=== [6/6] Install webhook auto-deploy service ==="
# Write environment file (webhook secret + paths)
cat > "$APP_DIR/deploy/webhook.env" <<ENV
WEBHOOK_SECRET=$WEBHOOK_SECRET
APP_DIR=$APP_DIR
DEPLOY_BRANCH=main
WEBHOOK_PORT=9000
ENV
chmod 600 "$APP_DIR/deploy/webhook.env"

sudo cp "$APP_DIR/deploy/vzebra-webhook.service" \
        /etc/systemd/system/vzebra-webhook.service
sudo systemctl daemon-reload
sudo systemctl enable vzebra-webhook
sudo systemctl restart vzebra-webhook
sleep 2
sudo systemctl is-active vzebra-webhook && echo "Webhook service running OK"

echo ""
echo "============================================================"
echo "  vzebra is live at https://$DOMAIN"
echo ""
echo "  Auto-deploy webhook:"
echo "  URL:    https://$DOMAIN/deploy/webhook"
echo "  Secret: $WEBHOOK_SECRET"
echo ""
echo "  Add in GitHub → Settings → Webhooks:"
echo "    Payload URL:  https://$DOMAIN/deploy/webhook"
echo "    Content type: application/json"
echo "    Secret:       $WEBHOOK_SECRET"
echo "    Events:       Just the push event"
echo "============================================================"
