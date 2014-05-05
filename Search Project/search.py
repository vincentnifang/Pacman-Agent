# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions

    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


class Node:
    def __init__(self, state, path, action, cost):
        self.state = state
        self.path = path
        self.action = action
        self.cost = cost

    def getState(self):
        return self.state

    def getPath(self):
        return self.path

    def getAction(self):
        return self.action

    def getCost(self):
        return self.cost


def search(problem, frontier, heur=None):
    startNode = Node(problem.getStartState(), [], None, 0)
    if problem.isGoalState(startNode.getState()):
        return []
    frontier.push(startNode)
    explored = []
    openset = {}
    openset[startNode.getState()] = 0

    while not frontier.isEmpty():
        node = frontier.pop()
        explored.append(node.getState())
        if problem.isGoalState(node.getState()):
            return node.getPath()
        for successor, direction, step in problem.getSuccessors(node.getState()):
            path = node.getPath() + [direction]
            cost = problem.getCostOfActions(path)
            if successor not in explored and (successor not in openset or openset[successor] > cost):
                frontier.push(Node(successor, path, direction, cost))
                openset[successor] = cost

    return []


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    # frontier = util.Stack()
    # return search(problem, frontier) """it's not work?"""

    startState = problem.getStartState()

    if problem.isGoalState(startState):
        return []
    frontier = util.Stack()
    frontier.push((startState, []))
    explored = []

    while not frontier.isEmpty():
        state, fatherpath = frontier.pop()
        explored.append(state)
        if problem.isGoalState(state):
            # print path
            return fatherpath
        for child, p, step in problem.getSuccessors(state):
            if child not in explored:
                path = fatherpath + [p]
                frontier.push((child, path))

    return []


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """

    frontier = util.Queue()
    return search(problem, frontier)

    # startState = problem.getStartState()
    #
    # if problem.isGoalState(startState):
    #     return []
    # frontier = util.Queue()
    # frontier.push((startState, []))
    # explored = [startState]
    #
    # while not frontier.isEmpty():
    #     state, fatherpath = frontier.pop()
    #     if problem.isGoalState(state):
    #         # print path
    #         return fatherpath
    #     for child, p, step in problem.getSuccessors(state):
    #         if child not in explored:
    #             path = fatherpath + [p]
    #             # if problem.isGoalState(child):
    #             #     # print path
    #             #     return path
    #             frontier.push((child, path))
    #             explored.append(child)
    #
    # return []
    #
    # util.raiseNotDefined()


def uniformCostSearch(problem):
    "Search the node of least total cost first. "

    def getCost(node):
        return problem.getCostOfActions(node.getPath())

    frontier = util.PriorityQueueWithFunction(getCost)
    return search(problem, frontier)


    # startState = problem.getStartState()
    # startCost = 0
    #
    # if problem.isGoalState(startState):
    #     return []
    # frontier = util.PriorityQueue()
    # frontier.push(startState, startCost)
    # explored = {startState: [[], startCost]}
    #
    # while not frontier.isEmpty():
    #     state = frontier.pop()
    #     if problem.isGoalState(state):
    #         return explored[state][0]
    #     if explored[state][0] == []:
    #         fatherpath = []
    #     else:
    #         fatherpath = explored[state][0]
    #     for child, p, step in problem.getSuccessors(state):
    #         childPath = fatherpath + [p]
    #         childCost = problem.getCostOfActions(childPath)
    #         if child not in explored.keys():
    #             frontier.push(child, childCost)
    #             explored[child] = [childPath, childCost]
    #         elif childCost < explored[child][1]:
    #             explored[child] = [childPath, childCost]
    #             frontier.push(child, childCost)
    #
    # return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."


    def getCost(node):
        return problem.getCostOfActions(node.getPath()) + heuristic(node.getState(), problem)

    frontier = util.PriorityQueueWithFunction(getCost)
    return search(problem, frontier)


    # startState = problem.getStartState()
    # gScore = 0
    # hScore = heuristic(startState, problem)
    # if problem.isGoalState(startState):
    #     return []
    # fScore = gScore + hScore
    # frontier = util.PriorityQueue()
    # frontier.push(startState, fScore)
    # explored = {str(startState): [[], fScore]}
    #
    # while not frontier.isEmpty():
    #     state = frontier.pop()
    #     if problem.isGoalState(state):
    #         return explored[str(state)][0]
    #
    #     if explored[str(state)][0] == []:
    #         fatherpath = []
    #     else:
    #         fatherpath = explored[str(state)][0]
    #
    #     for child, p, step in problem.getSuccessors(state):
    #         childPath = fatherpath + [p]
    #         gScore = problem.getCostOfActions(childPath)
    #         hScore = heuristic(child, problem)
    #         fScore = gScore + hScore
    #         childCost = fScore
    #         if str(child) not in explored.keys():
    #             frontier.push(child, childCost)
    #             explored[str(child)] = [childPath, childCost]
    #         elif childCost < explored[str(child)][1]:
    #             explored[str(child)] = [childPath, childCost]
    #             frontier.push(child, childCost)
    #
    # return []



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
