from stable_baselines3.common.env_checker import check_env

from bot.goleftenv import GoLeftEnv

def check():
    env = GoLeftEnv()
    # If the environment don't follow the interface, an error will be thrown
    check_env(env, warn=True)


def q():
    env = GoLeftEnv(grid_size=10)

    obs = env.reset()
    env.render()

    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())

    GO_LEFT = 0
    # Hardcoded best agent: always go left!
    n_steps = 20
    for step in range(n_steps):
        print("Step {}".format(step + 1))
        obs, reward, done, info = env.step(GO_LEFT)
        print('obs=', obs, 'reward=', reward, 'done=', done)
        env.render()
        if done:
            print("Goal reached!", "reward=", reward)
            break

if __name__=="__main__":
    check()
    q()