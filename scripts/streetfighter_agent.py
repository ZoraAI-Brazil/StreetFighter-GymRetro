import retro
import os


def main():
    env = retro.make(game='StreetFighterII-Genesis', state='round1')
    obs = env.reset()

    frameCount=0
    RyuWins = 0
    ChunLiWins = 0
    Roundcounted = False
    while True:
        action = env.action_space.sample()
        # 224 por 320
        obs, rew, done, info = env.step(action)

        os.system("clear")
        print(info)
        print("Vitórias Ryu(rounds): "+str(RyuWins)+"  Vitórias ChunLi(rounds): "+str(ChunLiWins))

        env.render()
        if done: #Everytime a round ends, both go life=0 for some fraction of frames, so we added the frameCounter
            frameCount += 1 #We need to env.reset just after 2/3 rounds, so we added this frameCounter to 30(amount of frames)
            Roundcounted = False
            if frameCount > 30:
                obs = env.reset()
                frameCount =0 #reset
        else:
            #print(info)
            if ((info["health"] > info["enemyHealth"]) and (info["enemyHealth"] <= 2) and Roundcounted == False):
                RyuWins += 1
                Roundcounted = True
            if ((info["health"] < info["enemyHealth"]) and (info["health"] < 1) and Roundcounted == False):
                ChunLiWins += 1
                Roundcounted = True




if __name__ == '__main__':
    main()
