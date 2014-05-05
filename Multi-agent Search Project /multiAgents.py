# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

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
    foodNum = 0

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        successorGameState = gameState.generatePacmanSuccessor(Directions.STOP)
        self.foodNum = len(successorGameState.getFood().asList())

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        print scores
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        # print legalMoves[chosenIndex]

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

        for ghost in newGhostStates:
            if util.manhattanDistance(newPos, ghost.getPosition()) < 2:
                return -99999

        newFoodLen = len(newFood.asList())

        if self.foodNum == 0:
            self.foodNum = len(newFood.asList())
        elif self.foodNum > len(newFood.asList()):
            return 999999

        return - min(util.manhattanDistance(newPos, food) for food in newFood.asList())

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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
        """

        maximum = float('-Inf')

        ghostNum = gameState.getNumAgents() - 1

        rootAction = Directions.STOP

        for action in gameState.getLegalActions(0):
            # print "cal pacman", action
            rootMax = self.__minValue(gameState.generateSuccessor(0, action), 1, ghostNum, self.depth)
            # print "rootMax", rootMax
            if rootMax > maximum:
                maximum = rootMax
                rootAction = action

        # print "pacman action is", rootAction
        # print "------------------------------------------------------------------------------"
        return rootAction


    def __maxValue(self, gameState, depth):
        ghostNum = gameState.getNumAgents() - 1
        depth -= 1
        # print "pacman", depth
        if depth == 0 or gameState.isLose() or gameState.isWin():
            # print "return value", self.evaluationFunction(gameState)
            return self.evaluationFunction(gameState)
        v = float('-Inf')
        for action in gameState.getLegalActions(0):
            # print "pacman to", action
            v = max(v, self.__minValue(gameState.generateSuccessor(0, action), 1, ghostNum, depth))

        # print "max", v

        return v

    def __minValue(self, gameState, agentIndex, ghostNum, depth):
        # print "ghost", agentIndex
        # print "depth", depth
        ghostNum -= 1
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        v = float('Inf')
        for action in gameState.getLegalActions(agentIndex):
            # print "ghost", agentIndex, "to", action
            # print "action and depth", action, depth
            # print "ghostNum", ghostNum
            if ghostNum > 0:
                v = min(v, self.__minValue(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, ghostNum,
                                           depth))
            else:
                # print "here"
                v = min(v, self.__maxValue(gameState.generateSuccessor(agentIndex, action), depth))

        # print "min", v
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):

        maximum = float('-Inf')

        ghostNum = gameState.getNumAgents() - 1

        rootAction = Directions.STOP

        alpha = float('-Inf')
        beta = float('Inf')

        for action in gameState.getLegalActions(0):
            # print "cal pacman", action
            rootMax = self.__minValue(gameState.generateSuccessor(0, action), 1, ghostNum, self.depth, alpha, beta)
            # print "rootMax", rootMax
            if rootMax > maximum:
                maximum = rootMax
                rootAction = action

            if maximum > beta:
                return maximum
            alpha = max(alpha, maximum)

        # print "pacman action is", rootAction
        # print "------------------------------------------------------------------------------"
        return rootAction


    def __maxValue(self, gameState, depth, alpha, beta):
        ghostNum = gameState.getNumAgents() - 1
        depth -= 1
        # print "pacman", depth
        if depth == 0 or gameState.isLose() or gameState.isWin():
            # print "return value", self.evaluationFunction(gameState)
            return self.evaluationFunction(gameState)
        v = float('-Inf')
        for action in gameState.getLegalActions(0):
            # print "pacman to", action
            v = max(v, self.__minValue(gameState.generateSuccessor(0, action), 1, ghostNum, depth, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)

        # print "max", v

        return v

    def __minValue(self, gameState, agentIndex, ghostNum, depth, alpha, beta):
        # print "ghost", agentIndex
        # print "depth", depth
        ghostNum -= 1
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        v = float('Inf')
        for action in gameState.getLegalActions(agentIndex):
            # print "ghost", agentIndex, "to", action
            # print "action and depth", action, depth
            # print "ghostNum", ghostNum
            if ghostNum > 0:
                v = min(v, self.__minValue(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, ghostNum,
                                           depth, alpha, beta))
            else:
                # print "here"
                v = min(v, self.__maxValue(gameState.generateSuccessor(agentIndex, action), depth, alpha, beta))

            if v < alpha:
                return v
            beta = min(beta, v)

        # print "min", v
        return v


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
        maximum = float('-Inf')

        ghostNum = gameState.getNumAgents() - 1

        rootAction = Directions.STOP

        for action in gameState.getLegalActions(0):
            # print "cal pacman", action
            rootMax = self.__expValue(gameState.generateSuccessor(0, action), 1, ghostNum, self.depth)
            # print "rootMax", rootMax
            if rootMax > maximum:
                maximum = rootMax
                rootAction = action

        # print "pacman action is", rootAction
        # print "------------------------------------------------------------------------------"
        return rootAction


    def __maxValue(self, gameState, depth):
        ghostNum = gameState.getNumAgents() - 1
        depth -= 1
        # print "pacman", depth
        if depth == 0 or gameState.isLose() or gameState.isWin():
            # print "return value", self.evaluationFunction(gameState)
            return float(self.evaluationFunction(gameState))
        v = float('-Inf')
        for action in gameState.getLegalActions(0):
            # print "pacman to", action
            v = max(v, self.__expValue(gameState.generateSuccessor(0, action), 1, ghostNum, depth))

        # print "max", v

        return v

    def __expValue(self, gameState, agentIndex, ghostNum, depth):
        ghostNum -= 1
        if gameState.isLose() or gameState.isWin():
            return float(self.evaluationFunction(gameState))
        v = float(0)
        actionNum = len(gameState.getLegalActions(agentIndex))
        # print gameState.getLegalActions(agentIndex)
        for action in gameState.getLegalActions(agentIndex):
            # print "ghost", agentIndex, "to", action
            # print "action and depth", action, depth
            # print "ghostNum", ghostNum
            if ghostNum > 0:
                vt = self.__expValue(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, ghostNum,
                                     depth)
            else:
                vt = self.__maxValue(gameState.generateSuccessor(agentIndex, action), depth)
            v += vt

        # print "min", v
        return v / actionNum


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      ***************
      *    point    *
      *      |      *
      *      |    F *
      *p     |      *
      ***************
      p = pacman, F= food, | is wall
      consider this state, you can not just calculate distance from p to F.
      you should calculate p to point add point to F
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    newWalls = currentGameState.getWalls().asList()

    foodNum = currentGameState.getNumFood()

    allScaredTimes = 0
    for st in newScaredTimes:
        allScaredTimes += st

    posGhostDistance = min(util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates)

    if posGhostDistance > 6:
        posGhostDistance = 6
    elif posGhostDistance < 3:
        posGhostDistance = -100

    if allScaredTimes > 0:
        posGhostDistance = 0

    if currentGameState.isLose():
        return -float('inf')
    elif currentGameState.isWin():
        # print "foodNum", foodNum
        return 5 * currentGameState.getScore() + (float)(posGhostDistance)



    # print "allScaredTimes", allScaredTimes



    # print "posGhostDistance", posGhostDistance

    def getPosFoodDistances(newPos, newFood):
        foodDists = map(lambda food: (food, util.manhattanDistance(newPos, food)), newFood.asList())

        minDist = float('inf')
        minFood = None

        for food, dist in foodDists:
            if dist < minDist:
                minDist = dist
                minFood = food

        return minFood, minDist

    minFood, posFoodDistance = getPosFoodDistances(newPos, newFood)
    # print "posFoodDistance", posFoodDistance

    def getDirection(newPos, minFood):
        return minFood[0] - newPos[0], minFood[1] - newPos[1]

    x, y = getDirection(newPos, minFood)

    def getWall(x, y):
        if x > 0:
            for i in range(1, x):
                wallx = newPos[0] + i
                wally = newPos[1] + int(i * y / x)
                wall = (wallx, wally)
                if wall in newWalls:
                    return wall
        if x == 0:
            wallx = newPos[0]
            for i in range(1, y):
                wally = newPos[1] + y
                wall = (wallx, wally)
                if wall in newWalls:
                    return wall
        if x < 0:
            for i in range(1, x):
                wallx = newPos[0] + i
                wally = newPos[1] + int(i * y / x)
                wall = (wallx, wally)
                if wall in newWalls:
                    return wall

        return None

    wall = getWall(x, y)

    def getYPoint(wall, x, y):
        if x > 0:
            for i in range(1, 100):
                if i % 2 == 0:
                    ix = -i / 2
                else:
                    ix = (i + 1) / 2
                if (wall[0], wall[1] + ix) not in newWalls:
                    return (wall[0] + i, wall[1])
        if x < 0:
            for i in range(1, 100):
                if i % 2 == 0:
                    ix = -i / 2
                else:
                    ix = (i + 1) / 2
                if (wall[0], wall[1] - ix) not in newWalls:
                    return (wall[0] + i, wall[1])

    if wall != None:
        point = getYPoint(wall, x, y)
        posFoodDistance = util.manhattanDistance(newPos, point) + util.manhattanDistance(point, minFood) + 2

    return (float)(posGhostDistance) - 0.1 * (float)(
        posFoodDistance) + 5 * currentGameState.getScore() - 10 * foodNum - 30 * len(
        currentGameState.getCapsules())

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

