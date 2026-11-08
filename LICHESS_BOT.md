# Run Bilari as a Lichess bot

The Lichess account upgrade succeeded (`ok: true`, HTTP 200). This only converts the account; keep the bot client running on your Mac for it to accept challenges and play. Bilari uses its puzzle-trained policy in pure Intuition mode: the UCI bridge selects a move directly from the policy and does not run search.

## 1. Update and install Bilari

In Terminal, from your existing Bilari clone:

```zsh
cd ~/Bilari
git pull
```

Use Python 3.11 or newer (Python 3.12 is recommended). Check the interpreter used by the environment:

```zsh
.venv/bin/python --version
```

If it reports Python 3.9 or the environment gives a Python path/`encodings` error, recreate the environment using a working Python 3.12 installation. This renames the old environment as a backup:

```zsh
mv .venv .venv-old
python3.12 -m venv .venv
```

Then install Bilari with pip updated. This uses a regular package install rather than editable mode:

```zsh
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install ".[train]"
```

If `python3.12` is not found, install Python 3.12 for macOS first, then repeat the environment creation step.

## 2. Install the Lichess connector

Follow the official [lichess-bot macOS installation guide](https://github.com/lichess-bot-devs/lichess-bot/wiki/How-to-Install) in a separate folder and Python environment. Copy its `config.yml.default` to `config.yml`.

In `config.yml`, set `token: "$LICHESS_BOT_TOKEN"` and merge the engine block from [`lichess-bot/engine-section.yml`](lichess-bot/engine-section.yml). Replace `/Users/yourname/Bilari` in that block with your Bilari folder path. The `dir` should end in `.venv/bin`; the engine name is `bilari-uci`.

## 3. Start the bot

In the same Terminal window, enter the Lichess OAuth token with the `bot:play` permission using hidden input, then start the connector from its own folder. Replace the final path with the folder where you installed `lichess-bot`:

```zsh
read -s "LICHESS_BOT_TOKEN?Paste your bot:play token (input hidden): "
printf '\n'
export LICHESS_BOT_TOKEN
cd /path/to/lichess-bot
venv/bin/python lichess-bot.py
```

Leave the Terminal window open while Bilari is online. Stop it with Control-C, then clear the token from that shell with `unset LICHESS_BOT_TOKEN`. Do not put the token in a file committed to GitHub or send it to anyone.

The official guide explains [OAuth token setup](https://github.com/lichess-bot-devs/lichess-bot/wiki/How-to-create-a-Lichess-OAuth-token) and [engine configuration](https://github.com/lichess-bot-devs/lichess-bot/wiki/Setup-the-engine).
