import magent
import pickle
import numpy as np
from magent.builtin.tf_model import DeepQNetwork
from magent.builtin.tf_model import DeepRecurrentQNetwork

if __name__ == "__main__":
    map_size = 100

    # init the game "pursuit"  (config file are stored in python/magent/builtin/config/)
    env = magent.GridWorld("battle", map_size=map_size)
    env.set_render_dir("build/render")

    # get group handles
    
    army1, army2 = env.get_handles()
    
    # init env and agents
    env.reset()
    env.add_walls(method="random", n=map_size * map_size * 0.02)
    env.add_agents(army1, method="random", n=map_size * map_size * 0.05)
    env.add_agents(army2,     method="random", n=map_size * map_size * 0.05)

    # init two models
    model1 = DeepQNetwork(env, army1, "battle-l", memory_size=2 ** 10)
    model2 = DeepQNetwork(env, army2, "battle-r", memory_size=2 ** 10)

    # load trained model
    model1.load("save_model", 9)
    model2.load("save_model", 10)

    done = False
    step_ct = 0
    print("nums: %d vs %d" % (env.get_num(army1), env.get_num(army2)))
    result = []
    while not done:
        # take actions for deers
        obs_1 = env.get_observation(army1)
        ids_1 = env.get_agent_id(army1)
        acts_1 = model1.infer_action(obs_1, ids_1)
        env.set_action(army1, acts_1)

        # take actions for tigers
        obs_2  = env.get_observation(army2)
        ids_2  = env.get_agent_id(army2)
        acts_2 = model2.infer_action(obs_2, ids_2)
        env.set_action(army2, acts_2)

        # simulate one step
        done = env.step()

        # render
        env.render()

        # get reward
        reward = [sum(env.get_reward(army1)), sum(env.get_reward(army2))]

        # clear dead agents
        env.clear_dead()

        # print info
        if step_ct % 10 == 0:
            print("step: %d\t predators' reward: %d\t preys' reward: %d" %
                    (step_ct, reward[0], reward[1]))
	result_tmp = []
	result_tmp.append(step_ct)
	result_tmp.append(reward[0])
	result_tmp.append(reward[1])
        result.append(result_tmp)
	step_ct += 1
        if step_ct > 1000:
            break
    result = np.array(result)
    np.save("./result/result_9_10.npy", result)
