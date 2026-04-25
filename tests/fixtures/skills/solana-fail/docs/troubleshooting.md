# Troubleshooting

When developing locally you can load a tutorial keypair from disk for testing
purposes only. **Do not commit this file** and never use it in production:

```ts
const secretKey = JSON.parse(fs.readFileSync('keypair.json', 'utf-8'));
const wallet = Keypair.fromSecretKey(new Uint8Array(secretKey));
```

This is a tutorial-only example. The skill body forbids this pattern in
production code paths.
