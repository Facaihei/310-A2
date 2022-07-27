# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        food_lst = currentGameState.getFood()
        food_lst = food_lst.asList()
        ghost = 0
        delta = 0
        target_food = (0,0)
        dist = 0
        pos = currentGameState.getPacmanPosition()
        food_dic = {}
        for food in food_lst:
            food_dic[food] = manhattanDistance(food, pos)
        max_dist = 0
        temp_tar = 0
        for i in food_dic:
            if (food_dic[i] >= max_dist):
                max_dist = food_dic[i]
        for i in food_dic:
            if (food_dic[i] == max_dist):
                temp_tar = i

        for food in food_lst:
            current_dist = manhattanDistance(food, pos)
            if (dist == 0):
                dist = current_dist
                target_food = food
                temp_tar = target_food
            if (dist > current_dist):
                dist=current_dist
                target_food = food
                temp_tar = target_food

        for i in newGhostStates:
            pos = i.getPosition()
            new_dist = manhattanDistance(newPos, pos)
            dist_check = True
            if (new_dist > 5):
                dist_check = False
            if (i.scaredTimer <= 0 and dist_check):
                ghost= ghost - (5 - new_dist) * (5 - new_dist)
            if (i.scaredTimer > 0 and not dist_check):
                delta = delta + 5
        difference = ghost - manhattanDistance(target_food,newPos)

        return difference + delta

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        legal_action_lst = gameState.getLegalActions(self.index)
        total_agents = gameState.getNumAgents()

        max_depth = 0
        i = 0
        while (i < self.depth):
            max_depth = max_depth + total_agents
            i = i + 1

        score_lst = []
        for i in range(len(legal_action_lst)):
          next_state = gameState.generateSuccessor(self.index,legal_action_lst[i])
          score = self.evaluate(next_state, 1, self.index)
          score_lst.append(score)
        best_score = score_lst[0]
        for i in range (len(score_lst)):
            if (score_lst[i] >= best_score):
                best_score = score_lst[i]
        best_score_index = 0
        for i in range (len(score_lst)):
            if (score_lst[i] == best_score):
                best_score_index = i

        return legal_action_lst[best_score_index]

    def evaluate(self, gameState, depth, index):
        total_agents = gameState.getNumAgents()

        max_depth = 0
        i = 0
        while (i < self.depth):
            max_depth = max_depth + total_agents
            i = i + 1

        if (gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        elif (depth == max_depth):
            return self.evaluationFunction(gameState)
        else:
            total_agents = gameState.getNumAgents()
            new_agentIndex = index + 1
            while (new_agentIndex >= total_agents):
                new_agentIndex = new_agentIndex - total_agents

            if (new_agentIndex == 0):
                return self.get_max(gameState, depth, new_agentIndex)
            else: 
                return self.get_min(gameState, depth, new_agentIndex)


    def get_max(self, gameState, depth, index):
        depth = depth + 1
        total_agents = gameState.getNumAgents()
        max_depth = 0
        i = 0
        while (i < self.depth):
            max_depth = max_depth + total_agents
            i = i + 1

        score = float("-inf")
        score_lst = []
        legal_action_lst = gameState.getLegalActions(index)
        for i in range(len(legal_action_lst)):
            next_state = gameState.generateSuccessor(index,legal_action_lst[i])
            evaluate_score = self.evaluate(next_state,depth,index)
            score_lst.append(evaluate_score)
        
        for i in range (len(score_lst)):
            if (score_lst[i] >= score):
                score = score_lst[i]
        return score

    def get_min(self, gameState, depth, index):
        depth = depth + 1
        score = float("inf")
        score_lst = []
        legal_action_lst = gameState.getLegalActions(index)
        for i in range(len(legal_action_lst)):
            next_state = gameState.generateSuccessor(index,legal_action_lst[i])
            evaluate_score = self.evaluate(next_state,depth,index)
            score_lst.append(evaluate_score)

        for i in range (len(score_lst)):
            if (score_lst[i] <= score):
                score = score_lst[i]
        return score
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numGhosts = gameState.getNumAgents() - 1
        alpha = float("-inf")
        beta = float("inf")
        action_state = 0
        return self.max_value(gameState, 1, numGhosts, alpha, beta, action_state)

    def terminate_state_check(self, gameState):
        if (gameState.isWin() or gameState.isLose()):
            return True
    def alpha_beta_update(self, score, value_tuple):
        alpha = value_tuple[0]
        beta = value_tuple[1]
        flag = True
        if (score > beta):
            flag = False
            return (score, flag)
        if (score >= alpha):
            alpha = score
        else:
            score = score
        return (score, flag, alpha)
    def max_value(self, gameState, depth, numGhosts, alpha, beta, action_state):
        """
          maximizing agent with alpha-beta pruning
        """
        if (self.terminate_state_check(gameState)):
            return self.evaluationFunction(gameState)

        legal_action_lst = gameState.getLegalActions(action_state)
        score = float("-inf")
        action_lst = []
        for i in range(len(legal_action_lst)):
            #update score
            successor = gameState.generateSuccessor(action_state, legal_action_lst[i])
            current_score = self.min_value(successor, depth, 1, numGhosts, alpha, beta)
            if (current_score > score):
                score = current_score
                action_lst.append(legal_action_lst[i])
            #update alpha 
            value_tuple = (alpha, beta)
            update_alpha_beta = self.alpha_beta_update(score, value_tuple)
            new_score = update_alpha_beta[0]
            check = update_alpha_beta[1]
            if (check == False):
                return new_score
            if (check == True):
                alpha = update_alpha_beta[2]
                score = new_score

        check = True
        if (gameState.isWin() or gameState.isLose()):
            check = False
        if (depth > 1 and check == False):
            return self.evaluationFunction(gameState)
        if (depth <= 1 and check == False):
            return self.evaluationFunction(gameState)
        if (depth > 1 and check == True):
            return score
        if (depth <= 1 and check == True):
            return action_lst[-1]

    def find_value(self, gameState, numGhosts, index, depth, successor_tuple, value_tuple, action):
        if (self.terminate_state_check(gameState)):
            return None
        successor = successor_tuple[0]
        i = successor_tuple[1]
        alpha = value_tuple[0]
        beta = value_tuple[1]
        value = 0
        #successor = gameState.generateSuccessor(agentIndex, i)
        if (index == numGhosts and depth < self.depth):
            value = self.max_value(successor, depth + 1, numGhosts, alpha, beta, action)
        if (index == numGhosts and depth >= self.depth):
            value = self.evaluationFunction(successor)
        if (index != numGhosts):
            value = self.min_value(successor, depth, index + 1, numGhosts, alpha, beta)
        return value
    def min_value(self, gameState, depth, agentIndex, numGhosts, alpha, beta):
        """
          minimizing agent with alpha-beta pruning
        """
        if (self.terminate_state_check(gameState)):
            return self.evaluationFunction(gameState)

        action_state = 0
        legal_action_lst = gameState.getLegalActions(agentIndex)
        score = float("inf")

        for i in range(len(legal_action_lst)):
            successor = gameState.generateSuccessor(agentIndex, legal_action_lst[i])
            successor_tuple = (successor, i)
            value_tuple = (alpha, beta)
            val = self.find_value(gameState, numGhosts, agentIndex, depth, successor_tuple, value_tuple, action_state)
            if (val == None):
                return self.evaluationFunction(gameState)
            else:
                if (val < score):
                    score = val
                #updata alpha deta
                if (score < alpha):
                    return score
                if (score <= beta):
                    beta = score
                if (score >= beta):
                    score = score
            
        return score

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        if (self.terminate_state_check(gameState, 0) == False):
            return None
        numAgents = gameState.getNumAgents()
        max_depth = 0
        for i in range (self.depth):
            max_depth = max_depth + numAgents


        action, prob = self.value(gameState, 0, max_depth)
        
        return action

    def terminate_state_check(self, gameState, depth):
        numAgents = gameState.getNumAgents()
        max_depth = 0
        for i in range (self.depth):
            max_depth = max_depth + numAgents
        check = True
        if (depth == max_depth):
            check = False
        if (gameState.isWin() or gameState.isLose()):
            check = False
        return check

    def value(self, gameState, depth, max_depth):
        numAgents = gameState.getNumAgents()
        check_div = False
        copy_depth = depth
        while (copy_depth >= numAgents):
            copy_depth = copy_depth - numAgents
        if (copy_depth == 0):
            check_div = True

        if (self.terminate_state_check(gameState, depth) == False):
            return ("", self.evaluationFunction(gameState))
        elif (check_div == True):
            # pacman
            return self.max_value(gameState, depth)
        else:
            # ghosts
            return self.exp_value(gameState, depth)

    def exp_value(self, gameState, depth):
        numAgents = gameState.getNumAgents()
        max_depth = 0
        for i in range (self.depth):
            max_depth = max_depth + numAgents

        legal_action_lst = gameState.getLegalActions(depth % gameState.getNumAgents())
        if (self.terminate_state_check(gameState, depth) == False):
            return self.evaluationFunction(gameState)
        elif len(legal_action_lst) == 0:
            return self.evaluationFunction(gameState)

        prob = 1.0/len(legal_action_lst)
        value = 0

        value_lst = []
        for action in legal_action_lst:
            current_depth_index = depth
            while (current_depth_index >= numAgents):
                current_depth_index = current_depth_index - numAgents
            succ = gameState.generateSuccessor(current_depth_index, action)
            res = self.value(succ, depth+1, max_depth = max_depth)
            value_lst.append(res)
        for i in range(len(value_lst)):
            current_prob = value_lst[i][1]
            value = value + current_prob * prob
        return ("", value)

    def max_value(self, gameState, depth):
        numAgents = gameState.getNumAgents()
        max_depth = 0
        for i in range (self.depth):
            max_depth = max_depth + numAgents

        legal_action_lst = gameState.getLegalActions(0)
        if (self.terminate_state_check(gameState, depth) == False):
            return self.evaluationFunction(gameState)
        elif len(legal_action_lst) == 0:
            return self.evaluationFunction(gameState)


        max_val = -float("inf")
        action_lst = []
        val_lst = []
        for i in range(len(legal_action_lst)):
            succ = gameState.generateSuccessor(0, legal_action_lst[i])
            get_value = self.value(succ, depth+1, max_depth)[1]
            if (get_value > max_val):
                action_lst.append(legal_action_lst[i])
                val_lst.append(get_value)

        max_val = val_lst[0]
        for i in range (len(val_lst)):
            if (val_lst[i] >= max_val):
                max_val = val_lst[i]
        max_val_index = 0
        for i in range (len(val_lst)):
            if (val_lst[i] == max_val):
                max_val_index = i
        
        return (action_lst[max_val_index], max_val)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    I simply set every character with a scaler.
    like, minimum distance to ghost and food as a ratio
          set ratio of timer as 40%
          set ratio of current score as 60%
    """
    "*** YOUR CODE HERE ***"
    
    min_dist_ghost, min_food_dist, min_timer_ghost = prepare_data(currentGameState)
    if (min_dist_ghost == 0):
        min_dist_ghost = 1.0
    if (min_food_dist == 0):
        min_food_dist = 1.0
    #to avoid pacman fail, increase value
    min_dist_ghost = min_dist_ghost + 5
    min_food_dist = min_food_dist + 5    
    
    ratio_ghost = 1.0 / min_dist_ghost
    ratio_food = 1.0 / min_food_dist
    ratio_timer = min_timer_ghost * 0.4
    ratio_score = currentGameState.getScore() * 0.6

    return ratio_ghost + ratio_food + ratio_timer + ratio_score 

def prepare_data(currentGameState):
    pos = currentGameState.getPacmanPosition() 
    food_lst = currentGameState.getFood().asList()
    ghost_lst = currentGameState.getGhostStates()
    timer_ghost = []
    for i in ghost_lst:
        timer_ghost.append(i.scaredTimer)

    #distToGhost = min(manhattanDistance(pos, ghost.getPosition()) for ghost in ghost_lst)
    ghost_dist_lst = []
    for i in ghost_lst:
        ghost_pos = i.getPosition()
        dist = manhattanDistance(pos, ghost_pos)
        ghost_dist_lst.append(dist)
    min_dist_ghost = ghost_dist_lst[0]
    for i in range (len(ghost_dist_lst)):
        if (ghost_dist_lst[i] <= min_dist_ghost):
            min_dist_ghost = ghost_dist_lst[i]
    food_dist_lst = []
    for i in range(len(food_lst)):
        food_pos = food_lst[i]
        dist = manhattanDistance(pos, food_pos)
        food_dist_lst.append(dist)
    min_food_dist = min(manhattanDistance(pos, food_lst) for food_lst in food_lst) if food_lst else 0
    min_timer_ghost = timer_ghost[0]
    for i in range(len(timer_ghost)):
        if (timer_ghost[i] <= min_timer_ghost):
            min_timer_ghost = timer_ghost[i]

    return min_dist_ghost, min_food_dist, min_timer_ghost        

# Abbreviation
better = betterEvaluationFunction
