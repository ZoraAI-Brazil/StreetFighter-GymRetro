# Street Fighter II Hyper Fighting bot with Gym-Retro

## Bases on OpenAi's Gym Retro

Gym Retro is a wrapper for video game emulator cores using the Libretro API to turn them into Gym environments.
It includes support for multiple classic game consoles and a dataset of different games.
It runs on Linux, macOS and Windows with Python 3.5 and 3.6 support.

Each game has files listing memory locations for in-game variables, reward functions based on those variables, episode end conditions, savestates at the beginning of levels and a file containing hashes of ROMs that work with these files.
Please note that ROMs are not included and you must obtain them yourself.

Currently supported systems:

- Atari 2600 (via Stella)
- Sega Genesis/Mega Drive (via Genesis Plus GX)

See [LICENSES.md](LICENSES.md) for information on the licenses of the individual cores.

## Installation

### Install Gym-Retro

Follow install instructions from OpenAI repository:

https://github.com/openai/retro

### Get a Street Fighter ROM

** DISCLAIMER: We don't take responsability for the origin of the ROM. You should buy Street Fighter II Hyper Fighting if you pretend to use the ROM. Any use of irregular/illegal ROMs is not on concern of the open-source organization ZoraAI and we take no responsability for it.

Supposing you have bought the game and have it's ROM, change the name of the file to 
```sh
rom.md
```

### Run Streetfighter_agent.py
