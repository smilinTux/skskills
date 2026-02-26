# Syncthing Troubleshooting

## Common Issues

### Devices Not Connecting

1. Check both devices have each other's Device ID added
2. Verify port 22000/TCP is open on both ends
3. Check `syncthing --verbose` for connection errors
4. Try disabling relays temporarily: Settings → Connections → Disable relays

### "Out of Sync" Items Won't Resolve

```bash
# Check for conflicts
find ~/synced-folder -name "*.sync-conflict-*"

# Force rescan
curl -X POST -H "X-API-Key: <key>" http://localhost:8384/rest/db/scan?folder=<id>
```

### High CPU / Memory Usage

- Reduce scan interval: `rescanIntervalS: 86400` (daily instead of hourly)
- Exclude large directories with `.stignore`
- Limit bandwidth: Settings → Connections → Rate Limits

### GUI Not Accessible

```bash
# Check if running
systemctl --user status syncthing

# Check the port
ss -tlnp | grep 8384

# Restart
systemctl --user restart syncthing
```
