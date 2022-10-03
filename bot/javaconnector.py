import random
import re
import os

import numpy as np
from py4j.java_gateway import JavaGateway, GatewayParameters

modifications = {'CAN_PAINT_1': 0, 'CAN_PAINT_2': 1, 'OTHER_PLAYER_CANNOT_PAINT': 2, 'DOUBLE_PAINT_RADIUS': 3,
                 'DOUBLE_SPEED': 4}
players_modifications = {'CAN_PAINT': 0, 'OTHER_PLAYER_CANNOT_PAINT': 1, 'DOUBLE_PAINT_RADIUS': 2,
                         'DOUBLE_SPEED': 3}

DEFAULT_JAVA_SERVER = "127.0.0.1"

class JavaConnector:
    def __init__(self, port):
        address = os.environ.get('JAVA_SERVER', DEFAULT_JAVA_SERVER)
        params = GatewayParameters(address=address, port=port)
        gateway = JavaGateway(gateway_parameters=params)
        self.controller = gateway.entry_point
        self.first_time = True
        self.prev_my_percent = 0
        self.prev_opp_percent = 0
        self.prev_my_mod_list = [0, ] * 4
        self.my_mod_list = [0, ] * 4
        self.opp_mod_list = [0, ] * 4
        self.can_paint = False
        self.game_info = None
        self.engine = None
        self.world = None
        self.summary = None
        self.player1 = None
        self.player2 = None
        self.my_player = None
        self.opp_player = None
        self.action_player = None
        self.my_player_ind = -1
        self.reset_season()
        print(self.action_player.getName(), self.action_player.getClass())
        self.my_player_ind = self.player1.getName() == 'Controlled' and 1 or 2

    def reset_season(self):
        self.game_info = None
        self.engine = None
        self.world = None
        self.summary = None
        while self.game_info is None:
            self.game_info = self.controller.getCurrentPlayingGame()
        while self.engine is None:
            self.engine = self.controller.getCurrentEngine()
        while self.world is None:
            self.world = self.controller.getCurrentWorld()
        while self.summary is None:
            self.summary = self.world.getWorldSummary()
        self.player1 = self.game_info.getPlayer1Implementation()
        self.player2 = self.game_info.getPlayer2Implementation()
        if self.player1.getName() == 'Controlled':
            self.my_player = self.player1.getMyPlayer(self.summary)
            self.opp_player = self.player1.getOtherPlayer(self.summary)
            self.action_player = self.player1
            self.my_player_ind = 1
        else:
            self.my_player = self.player1.getOtherPlayer(self.summary)
            self.opp_player = self.player1.getMyPlayer(self.summary)
            self.my_player_ind = 2
            self.action_player = self.player2

        self.first_time = True
        self.can_paint = False
        self.prev_my_mod_list = [0, ] * 4
        self.my_mod_list = [0, ] * 4
        self.opp_mod_list = [0, ] * 4
        self.prev_my_percent = 0
        self.prev_opp_percent = 0


    def make_one_step(self, action):
        action_player = self.action_player
        action_player.setAction_x(float(action[0]))
        action_player.setAction_y(float(action[1]))
        self.engine.setPy4j_can_do_turn(5)

    def get_observation(self, obs_size):
        self.can_paint = False
        obs = {}
        # mod_positions = [[0, 0]] * len(modifications)
        # summary = self.world.getWorldSummary()
        # cells = self.world.getCellsStr()
        cells = self.world.getCellsAroundPlayer('Controlled', obs_size, )
        cells = re.sub('], \\[', ', ', cells)
        cells = np.fromstring(cells[2:-2], sep=', ', dtype=int)
        positions = self.world.getPositionsPy4j()
        my_position = positions[self.my_player]  # 0..1000
        opp_position = positions[self.opp_player]
        my_position = (float(my_position.getX()/1000), float(my_position.getY()/1000))
        opp_position = (float(opp_position.getX()/1000), float(opp_position.getY()/1000))
        modificationPositions = self.summary.getModificationPositions()
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
        playersModification = self.summary.getPlayerModificationMap()
        self.my_mod_list = [0, ] * 4
        # self.opp_mod_list = [0, ] * 4
        my_mods = playersModification[self.my_player]
        # opp_mods = playersModification[self.opp_player]
        for m in my_mods:
            m_type = str(m.getType())
            self.my_mod_list[players_modifications[m_type]] = 1
            if m_type == "CAN_PAINT":
                self.can_paint = True
        # for m in opp_mods:
        #     m_type = str(m.getType())
        #     self.opp_mod_list[players_modifications[m_type]] = 1
        obs["grid"] = cells
        obs["positions"] = (my_position + opp_position)
        obs["modificators"] = mod_list
        # obs["mod_positions"] = mod_positions
        obs["my_mods"] = self.my_mod_list
        # obs["opp_mods"] = self.opp_mod_list
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
        # if reward > 0:
        #     reward *= 3
        if self.prev_opp_percent - opp_player_percent > 0:
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
            # print("Reset!")
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
                rew += 5
        self.prev_my_mod_list = self.my_mod_list
        # if rew > 0.5:
        #     print("Take modificator! ", rew)

        return rew
