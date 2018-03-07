import magent
from magent.builtin.tf_model import DeepQNetwork
from magent.builtin.tf_model import DeepRecurrentQNetwork
import pickle
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    map_size = 100
    results = []
    # init the game "pursuit"  (config file are stored in python/magent/builtin/config/)
    env = magent.GridWorld("battle", map_size=map_size)
    env.set_render_dir("build/render")

    # get group handles
    
    army1, army2, army3, army4, army5, army6, army7, army8, army9, army10 = env.get_handles()
    
    # init env and agents
    env.reset()
    env.add_walls(method="random", n=map_size * map_size * 0.02)
    env.add_agents(army1, method="random", n=map_size * map_size * 0.05)
    env.add_agents(army2, method="random", n=map_size * map_size * 0.05)
    env.add_agents(army3, method="random", n=map_size * map_size * 0.05)
    env.add_agents(army4, method="random", n=map_size * map_size * 0.05)
    env.add_agents(army5, method="random", n=map_size * map_size * 0.05)
    env.add_agents(army6, method="random", n=map_size * map_size * 0.05)
    env.add_agents(army7, method="random", n=map_size * map_size * 0.05)
    env.add_agents(army8, method="random", n=map_size * map_size * 0.05)
    env.add_agents(army9, method="random", n=map_size * map_size * 0.05)
    env.add_agents(army10, method="random", n=map_size * map_size * 0.05)

    # init two models
    # tf.reset_default_graph()
    model1 = DeepQNetwork(env, army1, "battle-l", memory_size=2**10)
    # tf.reset_default_graph()
    model2 = DeepQNetwork(env, army2, "battle-r", memory_size=2**10)
    model1.load("save_model", 18)
    model2.load("save_model", 18)

    # tf.reset_default_graph()
    model3 = DeepQNetwork(env, army3, "battle-l", memory_size=2**10)
    # tf.reset_default_graph()
    model4 = DeepQNetwork(env, army4, "battle-r", memory_size=2**10)
    model3.load("save_model", 17)
    model4.load("save_model", 17)
    
    # tf.reset_default_graph()
    model5 = DeepQNetwork(env, army5, "battle-l", memory_size=2**10)
    # tf.reset_default_graph()
    model6 = DeepQNetwork(env, army6, "battle-r", memory_size=2**10)
    model5.load("save_model", 16)
    model6.load("save_model", 16)

    # tf.reset_default_graph()
    model7 = DeepQNetwork(env, army7, "battle-l", memory_size=2**10)
    # tf.reset_default_graph()
    model8 = DeepQNetwork(env, army8, "battle-r", memory_size=2**10)
    model7.load("save_model", 15)
    model8.load("save_model", 15)

    # tf.reset_default_graph()
    model9 = DeepQNetwork(env, army9, "battle-l", memory_size=2**10)
    #tf.reset_default_graph()
    model10 = DeepQNetwork(env, army10, "battle-r", memory_size=2**10)
    model9.load("save_model", 14)
    model10.load("save_model", 14)

    done = False
    step_ct = 0
    print("nums: %d vs %d" % (env.get_num(army1), env.get_num(army2)))
    while not done:
        # take actions for army1
        obs_1 = env.get_observation(army1)
        ids_1 = env.get_agent_id(army1)
        acts_1 = model1.infer_action(obs_1, ids_1)
        env.set_action(army1, acts_1)

        # take actions for army2
        obs_2  = env.get_observation(army2)
        ids_2  = env.get_agent_id(army2)
        acts_2 = model2.infer_action(obs_2, ids_2)
        env.set_action(army2, acts_2)

        # take actions for army3
        obs_3  = env.get_observation(army3)
        ids_3  = env.get_agent_id(army3)
        acts_3 = model3.infer_action(obs_3, ids_3)
        env.set_action(army3, acts_3)

	# take actions for army4
        obs_4  = env.get_observation(army4)
        ids_4  = env.get_agent_id(army4)
        acts_4 = model4.infer_action(obs_4, ids_4)
        env.set_action(army4, acts_4)

	# take actions for army5
        obs_5  = env.get_observation(army5)
        ids_5  = env.get_agent_id(army5)
        acts_5 = model5.infer_action(obs_5, ids_5)
        env.set_action(army5, acts_5)

	# take actions for army6
        obs_6  = env.get_observation(army6)
        ids_6  = env.get_agent_id(army6)
        acts_6 = model6.infer_action(obs_6, ids_6)
        env.set_action(army6, acts_6)

	# take actions for army7
        obs_7  = env.get_observation(army7)
        ids_7  = env.get_agent_id(army7)
        acts_7 = model7.infer_action(obs_7, ids_7)
        env.set_action(army7, acts_7)

	# take actions for army8
        obs_8  = env.get_observation(army8)
        ids_8  = env.get_agent_id(army8)
        acts_8 = model8.infer_action(obs_8, ids_8)
        env.set_action(army8, acts_8)

	# take actions for army9
        obs_9  = env.get_observation(army9)
        ids_9  = env.get_agent_id(army9)
        acts_9 = model9.infer_action(obs_9, ids_9)
        env.set_action(army9, acts_9)

	# take actions for army10
        obs_10  = env.get_observation(army10)
        ids_10  = env.get_agent_id(army10)
        acts_10 = model10.infer_action(obs_10, ids_10)
        env.set_action(army10, acts_10)

        # simulate one step
        done = env.step()

        # render
        env.render()

        # get reward
        reward = [sum(env.get_reward(army1)), sum(env.get_reward(army2)),sum(env.get_reward(army3)),sum(env.get_reward(army4)),sum(env.get_reward(army5)),sum(env.get_reward(army6)),sum(env.get_reward(army7)),sum(env.get_reward(army8)),sum(env.get_reward(army9)),sum(env.get_reward(army10))]
	results.append(reward)
        # clear dead agents
        env.clear_dead()

        # print info
        if step_ct % 10 == 0:
            print("step: %d\t army1' reward: %d\t army2' reward: %d" %
                    (step_ct, reward[0], reward[1]))
	    print("step: %d\t army3' reward: %d\t army4' reward: %d" %
                    (step_ct, reward[2], reward[3]))
	    print("step: %d\t army5' reward: %d\t army6' reward: %d" %
                    (step_ct, reward[4], reward[5]))
	    print("step: %d\t army7' reward: %d\t army8' reward: %d" %
                    (step_ct, reward[6], reward[7]))
	    print("step: %d\t army9' reward: %d\t army10' reward: %d" %
                    (step_ct, reward[8], reward[9]))

        step_ct += 1
        if step_ct > 1000:
	    result = np.array(result)
	    np.save("./result/result_battle.npy", result)
            break


