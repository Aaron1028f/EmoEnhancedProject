### how to run (deprecated now)

```bash
conda activate frontend-env

# template source: https://github.com/livekit-examples/agent-starter-react
# lk app create --template agent-starter-react
/home/aaron/project/server/lk_exp/server_src_bin/lk app create --template agent-starter-react
cd /home/aaron/project/client/lk_ui_from_template

# npm install -g pnpm

pnpm install

# run the app
pnpm dev

```

### not used now
```bash
# # source: https://docs.livekit.io/home/quickstarts/react/
# npm install @livekit/components-react @livekit/components-styles livekit-client --save

# # create the new project
# npm create vite@latest

# > npx
# > "create-vite"

# │
# ◇  Project name:
# │  lk_webui
# │
# ◇  Select a framework:
# │  React
# │
# ◇  Select a variant:
# │  TypeScript
# │
# ◇  Scaffolding project in /home/aaron/project/client/lk_webui...
# │
# └  Done. Now run:

#   cd lk_webui
#   npm install
#   npm run dev
```