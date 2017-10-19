#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd

dir = "../corpus/TBBT/"
trans_dir = "NE_1st2nd3rd/"
scene_dir = "scenes/"
res_dir = "../data/"

if __name__ == '__main__':
    # -----Read the time information of the scene of all episode, it's a list of list of tuple----- #
    # [
    # [(episode1_scene_1_start, episode1_scene1_end), (episode1_scene_2_start, episode1_scene2_end)],
    # [(episode2_scene_1_start, episode2_scene1_end), (episode2_scene_2_start, episode2_scene2_end)]
    # ]
    time_list = []
    for i in range(12):
        if i < 9:
            file =  dir + scene_dir + "tbbt.season01.episode" + str(0) + str(i + 1) + ".scenes.txt"
        else:
            file = dir + scene_dir + "tbbt.season01.episode" + str(i + 1) + ".scenes.txt"
        with open(file, 'r') as f:
            tmp = []
            while 1:
                line = f.readline()
                if not line:
                    break
                else:
                    line = line[:-1].split(' ')
                    tmp.append((line[1], line[2]))
            time_list.append(tmp)
    # print(time_list)

    # -----Read the translation of all episode, like a list of DataFrame----- #
    # [DF1, DF2, DF3]