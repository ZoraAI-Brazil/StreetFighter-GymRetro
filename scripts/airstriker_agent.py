import retro
def main():
    env = retro.make(game='Airstriker-Genesis', state='Level1')
    obs = env.reset()
    while True:
        action = env.action_space.sample()

        obs, rew, done, info = env.step(action)


        print(info)
        env.render()
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()
