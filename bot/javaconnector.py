import random
import re
import time

import numpy as np
from py4j.java_gateway import JavaGateway, GatewayParameters

modifications = {'CAN_PAINT_1': 0, 'CAN_PAINT_2': 1, 'OTHER_PLAYER_CANNOT_PAINT': 2, 'DOUBLE_PAINT_RADIUS': 3,
                 'DOUBLE_SPEED': 4}
players_modifications = {'CAN_PAINT': 0, 'OTHER_PLAYER_CANNOT_PAINT': 1, 'DOUBLE_PAINT_RADIUS': 2,
                         'DOUBLE_SPEED': 3}


class JavaConnector:
    def __init__(self, port):
        params = GatewayParameters(port=port)
        gateway = JavaGateway(gateway_parameters=params)
        self.controller = gateway.entry_point
        self.reset_season()
        self.first_time = True
        self.prev_my_percent = 0
        self.prev_opp_percent = 0
        self.prev_my_mod_list = [0, ] * 4
        self.my_mod_list = [0, ] * 4
        self.opp_mod_list = [0, ] * 4
        self.can_paint = False

    def reset_season(self):
        self.controller.startSeason()
        self.first_time = True
        self.can_paint = False
        self.prev_my_mod_list = [0, ] * 4
        self.my_mod_list = [0, ] * 4
        self.opp_mod_list = [0, ] * 4
        self.prev_my_percent = 0
        self.prev_opp_percent = 0
        time.sleep(1)
        self.game_info = self.controller.getCurrentPlayingGame()
        self.engine = self.controller.getEngine()
        self.world = self.controller.getWorld()
        self.player1 = self.game_info.getPlayer1()
        self.player2 = self.game_info.getPlayer2()
        self.my_player_ind = self.player1.getName() == "ML" and 1 or 2
        if self.player1.getName() == "ML":
            self.my_player = self.player1
            self.opp_player = self.player2
        else:
            self.my_player = self.player2
            self.opp_player = self.player1
        # print(self.my_player)

    def make_one_step(self, action):
        # print("step")
        my_player = self.my_player
        my_player.setAction_x(float(action[0] * 100))
        my_player.setAction_y(float(action[1] * 100))
        self.engine.setExternal_do_turn(True)
        # i = 0
        # while self.engine.getNot_breakable_counter() > 0:
        #     time.sleep(0.1)
        #     # print(i)
        #     i += 1
        #     continue
        # engine.setExternal_control(False)

    def get_observation(self, obs_size):
        self.can_paint = False
        obs = {}
        mod_positions = [[0, 0]] * len(modifications)
        summary = self.world.getWorldSummary()
        # cells = self.world.getCellsStr()
        cells = self.world.getCellsPlayer("ML", obs_size, )
        cells = re.sub('], \\[', ', ', cells)
        cells = np.fromstring(cells[2:-2], sep=', ', dtype=int)
        positions = summary.getPlayerPositions()
        my_position = positions[self.my_player]  # 0..1000
        opp_position = positions[self.opp_player]
        my_position = (int(my_position.getX()/100), int(my_position.getY()/100))
        opp_position = (int(opp_position.getX()/100), int(opp_position.getY()/100))
        modificationPositions = summary.getModificationPositions()
        mod_list = [0, ] * 5
        for m in modificationPositions:
            m_type = str(m.getType())
            if m_type == 'CAN_PAINT':
                if modificationPositions[m].getX() == 100:
                    m_type += '_1'
                else:
                    m_type += '_2'
            mod_list[modifications[m_type]] = 1
            # mod_positions[modifications[m_type]] = (int(modificationPositions[m].getX()/10), int(modificationPositions[m].getY()/10))
        playersModification = summary.getPlayerModificationMap()
        self.my_mod_list = [0, ] * 4
        self.opp_mod_list = [0, ] * 4
        my_mods = playersModification[self.my_player]
        opp_mods = playersModification[self.opp_player]
        for m in my_mods:
            m_type = str(m.getType())
            self.my_mod_list[players_modifications[m_type]] = 1
            if m_type == "CAN_PAINT":
                self.can_paint = True
        for m in opp_mods:
            m_type = str(m.getType())
            self.opp_mod_list[players_modifications[m_type]] = 1
        obs["grid"] = cells
        obs["positions"] = (my_position, opp_position)
        obs["modificators"] = mod_list
        # obs["mod_positions"] = mod_positions
        obs["my_mods"] = self.my_mod_list
        obs["opp_mods"] = self.opp_mod_list
        # print(time.time(), obs.values())
        return obs

    def get_reward(self):
        reward = 0
        player1_percent = self.game_info.getPlayer1Percent()
        player2_percent = self.game_info.getPlayer2Percent()
        if self.my_player_ind == 1:
            my_player_percent = player1_percent
            opp_player_percent = player2_percent
        else:
            my_player_percent = player2_percent
            opp_player_percent = player1_percent

        reward += my_player_percent - self.prev_my_percent
        if reward > 0:
            reward *= 3
        # if self.prev_opp_percent - opp_player_percent > 0:
        reward += self.prev_opp_percent - opp_player_percent
        # if self.can_paint:
        #     if my_player_percent <= self.prev_my_percent:
        #         # print("old cell not painted")
        #         reward -= 0.1
        #     else:
        #         reward += 0.1
        #         # print("new cell painted")
        self.prev_my_percent = my_player_percent
        self.prev_opp_percent = opp_player_percent
        reward += self.calc_mods()
        # if reward > 0:
        #     print("base reward %s" % reward)
        if self.first_time:
            self.first_time = False
            print("Reset!")
            return 0
        # print(reward)
        return reward

    def is_round_over(self):
        return self.game_info.isFinished()

    def calc_mods(self):
        rew = 0
        # rew = sum(self.my_mod_list) * 0.05
        # if rew >= 0.1:
        #     print("double buf! ", self.my_mod_list)
        for i in range(len(self.my_mod_list)):
            if self.my_mod_list[i] - self.prev_my_mod_list[i] == 1:
                rew += 10
        self.prev_my_mod_list = self.my_mod_list
        # if rew > 0.5:
        #     print("Take modificator! ", rew)

        return rew
