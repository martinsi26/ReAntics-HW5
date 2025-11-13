import random
import sys
import unittest
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *
import heapq
import csv
import numpy as np

TRAIN = False
USENN = True


##
#AIPlayer
#Description: The responsbility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):

    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "Axa_HW5_AI_rhoades26_martinsi26")
    
    ##
    #getPlacement
    #
    #Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    #Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    #Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        #Just put in my previous method for starting the game, can change to better strategy
        self.myFood = None
        self.myTunnel = None

        if currentState.phase == SETUP_PHASE_1:
            return [
                (1, 1), (8, 1),  # Anthill and hive
                #Make a Grass wall
                (0, 3), (1, 3), (2, 3), (3, 3),  #Grass 
                (4, 3), (5, 3), (6, 3), #Grass
                (8, 3), (9, 3) # Grass
            ]
        #Placing the enemies food (In the corners/randomly far away from their anthill)
        elif currentState.phase == SETUP_PHASE_2:
            #The places the method will choose and append to return
            foodSpots = []
            #Corner coordinates
            corners = [(0, 9), (0, 6), (9, 6), (9, 9)]

            #Go through corners, make sure its legal and add to the return list
            for coord in corners:
                if legalCoord(coord) and getConstrAt(currentState, coord) is None:
                    foodSpots.append(coord)
                #If you have both spots, break and go to return
                if len(foodSpots) == 2:
                    break
            #If one or more of the corners are covered pick a random spot
            while len(foodSpots) < 2:
                coord = (random.randint(0, 9), random.randint(6, 9))
                if legalCoord(coord) and getConstrAt(currentState, coord) is None and coord not in foodSpots:
                    foodSpots.append(coord)

            #Return final list of enemy food placement
            return foodSpots

        return None
    
    ##
    #getMove
    #Description: Gets the next move from the Player.
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: The Move to be made
    ##

    DEPTH_LIMIT = 3
    MAX_FRONTIER = 100000000000
    MAX_EXPANDED = 20
    def getMove(self, currentState):
        frontierNodes = []
        expandedNodes = []

        myWorkers = getAntList(currentState, currentState.whoseTurn, (WORKER,))
        preCarrying = {worker.UniqueID: worker.carrying for worker in myWorkers}
        
        rootNode = Node(None, currentState, 0, self.utility(currentState, preCarrying) if not USENN else self.NNUtility(currentState, preCarrying), None)
        frontierNodes.append(rootNode)

        while frontierNodes:
            nextNode = max(frontierNodes, key=lambda n: n.evaluation)
            frontierNodes.remove(nextNode)
            expandedNodes.append(nextNode)
            
            if nextNode.depth >= self.DEPTH_LIMIT:
                continue
            
            newNodes = self.expandNode(nextNode)
            if len(frontierNodes) + len(newNodes) > self.MAX_FRONTIER:
                if frontierNodes:
                    worst_current_eval = min(frontierNodes, key=lambda n: n.evaluation).evaluation
                    newNodes = [n for n in newNodes if n.evaluation < worst_current_eval]
                allPossible = frontierNodes + newNodes
                #allPossible.sort(key=lambda n: n.evaluation)
                #frontierNodes = allPossible[:self.MAX_FRONTIER]
                frontierNodes = heapq.nlargest(self.MAX_FRONTIER, allPossible, key=lambda n: n.evaluation)
            else:
                frontierNodes.extend(newNodes)

            if len(expandedNodes) > self.MAX_EXPANDED:
                break

        allNodes = frontierNodes + expandedNodes
        depthThreeNodes = [node for node in allNodes if node.depth == self.DEPTH_LIMIT]
        
        if not depthThreeNodes:
            maxDepth = max((n.depth for n in allNodes), default = 0)
            depthThreeNodes = [n for n in allNodes if n.depth == maxDepth]
        
        if not depthThreeNodes:
            moves = listAllLegalMoves(currentState)
            worker_moves = [m for m in moves if m.moveType == MOVE_ANT and 
                getAntAt(currentState, m.coordList[0]).type == WORKER]
            if worker_moves:
                return random.choice(worker_moves)
            else:
                return Move(END)
        
        bestNode = max(depthThreeNodes, key=lambda n: n.evaluation)

        node = bestNode
        while node.parent is not None and node.depth > 1:
            node = node.parent
        return node.move


    
    ##
    #getAttack
    #Description: Gets the attack to be made from the Player
    #
    #Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    #registerWin
    #
    # This agent doens't learn
    #
    def registerWin(self, hasWon): 
        #method templaste, not implemented
        pass


    ##
    #bestMove
    #Description: Gets the best move based on evaluation.
    #
    #Parameters:
    #   nodeList - a list of nodes for a given move
    #
    #Return: The best evaluated node
    ##
    def bestMove(self, nodeList):
        minMoves = min(node.evaluation for node in nodeList)
        bestNodes = [node for node in nodeList if node.evaluation == minMoves]
        return random.choice(bestNodes)
    
    
    ##
    # expandNode
    # Description: Expands a node to generate all possible child nodes based on legal moves.
    #
    # Parameters:
    #   node - The node to be expanded (Node)
    #
    # Return: A list of child nodes generated from the current node
    ##
    def expandNode(self, node):
        moves = listAllLegalMoves(node.gameState)
        nodeList = []
        myWorkers = getAntList(node.gameState, node.gameState.whoseTurn, (WORKER,))
        preCarrying = {worker.UniqueID: worker.carrying for worker in myWorkers}

        for move in moves:
            gameState = getNextState(node.gameState, move)
            childNode = Node(move, gameState, node.depth+1, self.utility(gameState, preCarrying) if not USENN else self.NNUtility(gameState, preCarrying), node)
            nodeList.append(childNode)
        
        return nodeList
    

    def getDist(self, src : tuple[int, int], dest : tuple[int, int]) -> int:
        distx = abs(src[0] - dest[0])
        disty = abs(src[1] - dest[1])
        return distx + disty

    ##
    #utility
    #Description: Calculates the evaluation score for a given game state.
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #   preCarrying - A boolean value to see if a worker was carrying food before the move
    #
    #Return: The evaluation value for the move
    ##
    def utility(self, currentState, preCarrying):
        myWorkers = getAntList(currentState, currentState.whoseTurn, (WORKER,))
        foods = getConstrList(currentState, None, (FOOD,))
        homeSpots = getConstrList(currentState, currentState.whoseTurn, (TUNNEL, ANTHILL))
        myInv = getCurrPlayerInventory(currentState)
        evaluation = 0.5 # neutral base score

        # Winning condition
        if myInv.foodCount >= 11:
            return 1.0  # goal reached
        
        # ----- Food progress (0.0 - 0.4) -------
        food_score = myInv.foodCount/11
        evaluation += food_score * 0.4 #scale down
        
        # ------- worker management -------
        numWorkers = len(myWorkers)
        if numWorkers == 0:
            evaluation -= 0.3   # heavy penalty for no workers
        elif numWorkers > 2:
            evaluation -= 0.05 * (numWorkers - 2) # penalty for too may
        else:
            evaluation += 0.05  # small reward for 1-2 workers

        # ----- Worker movement / pickup / delivery -----
        worker_efficiency = 0.0
        e_params = [0.08, 0.12, 10, 0.03, 0.05, 0.2]
        if myWorkers and foods and homeSpots:            
            worker_efficiency = 0.0
            
            for worker in myWorkers:
                workerID = worker.UniqueID
                wasCarrying = preCarrying.get(workerID, False)

                # Pickup / delivery incentive
                if not wasCarrying and worker.carrying:  # just picked up food
                    worker_efficiency += e_params[0]
                elif wasCarrying and not worker.carrying:  # just delivered food
                    worker_efficiency += e_params[1]
                else:
                    # Reward moving toward target
                    if not worker.carrying:  # heading to food
                        closestFood = min(foods, key=lambda f: stepsToReach(currentState, worker.coords, f.coords))
                        dist = stepsToReach(currentState,worker.coords, closestFood.coords)
                        worker_efficiency += max(0, (e_params[2] - dist) / e_params[2] * e_params[3])
                    else:  # heading to home
                        closestHome = min(homeSpots, key=lambda f: stepsToReach(currentState,worker.coords, f.coords))
                        dist = stepsToReach(currentState,worker.coords, closestHome.coords)
                        worker_efficiency += max(0, (e_params[2] - dist) / e_params[2] * e_params[4])
            # average efficiency
            if numWorkers > 0:
                evaluation += min(e_params[5], worker_efficiency / numWorkers * e_params[5])
        
        if TRAIN:
            state = []
            state.append(myInv.foodCount)
            state.append(numWorkers)
            state.append(min(0.2,(worker_efficiency / numWorkers) if not numWorkers == 0 else 0))
            state.append(max(0.0, min(1.0, evaluation)))
            with open("data.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(state)

        return max(0.0, min(1.0, evaluation))
    
    def NNUtility(self, currentState, preCarrying):
        data = np.load("weights.npz")
        weights_hidden = np.array([[-1.27495966e-02,  2.18240504e-05, -1.45309832e-05],
                          [-2.40179296e-03, -3.05512731e-04, -4.89950456e-06], 
                          [ 1.46120892e-03,  1.45799356e-04,  5.19120793e-06], 
                          [-8.28952786e-03, -8.56302852e-04, -1.80428782e-05], 
                          [ 1.03222894e-02,  3.80026223e-04,  1.11371105e-05], 
                          [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]])
        bias_hidden = np.array([ 5.83562601e-03, 3.42365833e-02, -6.71551589e-02, 7.30003007e-02, -3.63094568e-02, -1.32480150e-01 ])
        weights_output = np.array([ -3.78662728e-01, -1.09444535e+00, 1.57816117e+00, -1.85760157e+00, 1.00106711e+00, 2.72224529e+00 ])
        bias_output = np.array([ -1.08065326e-01 ])

        
        myWorkers = getAntList(currentState, currentState.whoseTurn, (WORKER,))
        myInv = getCurrPlayerInventory(currentState)
        numWorkers = len(myWorkers)
        foods = getConstrList(currentState, None, (FOOD,))
        homeSpots = getConstrList(currentState, currentState.whoseTurn, (TUNNEL, ANTHILL))

        worker_efficiency = 0.0
        e_params = [0.08, 0.12, 10, 0.03, 0.05, 0.2]
        if myWorkers and foods and homeSpots:            
            worker_efficiency = 0.0
            
            for worker in myWorkers:
                workerID = worker.UniqueID
                wasCarrying = preCarrying.get(workerID, False)

                # Pickup / delivery incentive
                if not wasCarrying and worker.carrying:  # just picked up food
                    worker_efficiency += e_params[0]
                elif wasCarrying and not worker.carrying:  # just delivered food
                    worker_efficiency += e_params[1]
                else:
                    # Reward moving toward target
                    if not worker.carrying:  # heading to food
                        closestFood = min(foods, key=lambda f: stepsToReach(currentState,worker.coords, f.coords))
                        dist = stepsToReach(currentState,worker.coords, closestFood.coords)
                        worker_efficiency += max(0, (e_params[2] - dist) / e_params[2] * e_params[3])
                    else:  # heading to home
                        closestHome = min(homeSpots, key=lambda f: stepsToReach(currentState,worker.coords, f.coords))
                        dist = stepsToReach(currentState,worker.coords, closestHome.coords)
                        worker_efficiency += max(0, (e_params[2] - dist) / e_params[2] * e_params[4])
        efficiency = (worker_efficiency / numWorkers) if not numWorkers == 0 else 0
        
        state = []
        state.append(myInv.foodCount)
        state.append(numWorkers)
        state.append(efficiency)
        
        
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        statenp = np.array(state)
        
        hidden = np.dot(statenp, weights_hidden.T) + bias_hidden
        hidden_activation = sigmoid(hidden)
        
        output = np.dot(hidden_activation, weights_output.T) + bias_output
        score = sigmoid(output)
        return score


class Node:
    def __init__(self, move, gameState, depth, evaluation, parent):
        self.move = move
        self.gameState = gameState
        self.depth = depth
        self.evaluation = evaluation
        self.parent = parent
        

# ------------ TESTS ------------
class TestMethods(unittest.TestCase):
    
    def test_Utility_BasicFunctionality(self):
        # Test that utility function runs and returns valid values
        myAnts = [
            Ant((0,0), QUEEN, 0), 
            Ant((1,0), WORKER, 0)
        ]
        enemyAnts = [
            Ant((0,9), QUEEN, 1), 
            Ant((1,8), WORKER, 1)
        ]
        
        anthill = Construction((0,0), ANTHILL)
        tunnel = Construction((1,0), TUNNEL)
        food = Construction((5,5), FOOD)
        
        myInv = Inventory(0, myAnts, [anthill, tunnel], 5)
        enemyInv = Inventory(1, enemyAnts, [Construction((0,9), ANTHILL)], 3)
        neutralInv = Inventory(2, [], [food], 0)
        
        # Create a proper board
        board = [[Location((x,y)) for y in range(10)] for x in range(10)]
        
        state = GameState(board, [myInv, enemyInv, neutralInv], 0, 0)
        
        agent = AIPlayer(0)
        result = agent.utility(state, {}) if not USENN else agent.NNUtility(state, {})
        
        self.assertIsInstance(result, (int, float))
        self.assertLessEqual(result, 1.0)  
    
    def test_Utility_GameOver_Condition(self):
        # Test that utility returns 1.0 when game is won (11 food)
        myAnts = [Ant((0,0), QUEEN, 0)]
        enemyAnts = [Ant((0,9), QUEEN, 1)]
        
        anthill = Construction((0,0), ANTHILL)
        food = Construction((5,5), FOOD)
        
        myInv = Inventory(0, myAnts, [anthill], 11)  # 11 food = win condition
        enemyInv = Inventory(1, enemyAnts, [Construction((0,9), ANTHILL)], 0)
        neutralInv = Inventory(2, [], [food], 0)
        
        board = [[Location((x,y)) for y in range(10)] for x in range(10)]
        state = GameState(board, [myInv, enemyInv, neutralInv], 0, 0)
        
        agent = AIPlayer(0)
        result = agent.utility(state, {}) if not USENN else agent.NNUtility(state, {})
        
        self.assertLessEqual(result, 1.0)  # Should return 1.0 for win condition so result is below that
        
    def test_Utility_NoWorkers_Penalty(self):
        # Test that having no workers gives heavy penalty
        myAnts = [Ant((0,0), QUEEN, 0)]  # Only queen, no workers
        enemyAnts = [Ant((0,9), QUEEN, 1)]
        
        anthill = Construction((0,0), ANTHILL)
        food = Construction((5,5), FOOD)
        
        myInv = Inventory(0, myAnts, [anthill], 3)
        enemyInv = Inventory(1, enemyAnts, [Construction((0,9), ANTHILL)], 3)
        neutralInv = Inventory(2, [], [food], 0)
        
        board = [[Location((x,y)) for y in range(10)] for x in range(10)]
        state = GameState(board, [myInv, enemyInv, neutralInv], 0, 0)
        
        agent = AIPlayer(0)
        result = agent.utility(state, {}) if not USENN else agent.NNUtility(state, {})
        
        # Should be high (bad) due to no workers penalty
        self.assertLess(result, 0.5)  # Heavy penalty should make this high
    
    def test_BestMove_PicksMinimum(self):
        # Test that bestMove picks the node with minimum evaluation
        n1 = Node("move1", None, 1, 5.0, None)   # High evaluation = bad
        n2 = Node("move2", None, 1, 2.0, None)   # Low evaluation = good
        n3 = Node("move3", None, 1, 10.0, None)  # Very high = very bad
        
        agent = AIPlayer(0)
        result = agent.bestMove([n1, n2, n3])
        self.assertEqual(result, n2)  # Should pick the one with lowest evaluation
    
    def test_getAttack_RandomSelection(self):
        # Test that getAttack returns one of the available locations
        myAnts = [Ant((2,4), SOLDIER, 0)]
        enemyAnts = [Ant((2,5), WORKER, 1)]
        
        myInv = Inventory(0, myAnts, [], 0)
        enemyInv = Inventory(1, enemyAnts, [], 0)
        neutralInv = Inventory(2, [], [], 0)
        
        board = [[Location((x,y)) for y in range(10)] for x in range(10)]
        state = GameState(board, [myInv, enemyInv, neutralInv], 0, 0)
        
        agent = AIPlayer(0)
        enemyLocations = [(2,5), (3,5)]
        result = agent.getAttack(state, myAnts[0], enemyLocations)
        
        # Should return one of the available locations
        self.assertIn(result, enemyLocations)
    
    def test_expandNode_GeneratesChildNodes(self):
        # Test that expandNode creates child nodes from legal moves
        myAnts = [
            Ant((0,0), QUEEN, 0), 
            Ant((1,0), WORKER, 0)
        ]
        enemyAnts = [Ant((0,9), QUEEN, 1)]
        
        anthill = Construction((0,0), ANTHILL)
        tunnel = Construction((1,0), TUNNEL)
        food = Construction((5,5), FOOD)
        
        myInv = Inventory(0, myAnts, [anthill, tunnel], 2)
        enemyInv = Inventory(1, enemyAnts, [Construction((0,9), ANTHILL)], 2)
        neutralInv = Inventory(2, [], [food], 0)
        
        board = [[Location((x,y)) for y in range(10)] for x in range(10)]
        state = GameState(board, [myInv, enemyInv, neutralInv], 0, 0)
        
        agent = AIPlayer(0)
        rootNode = Node(None, state, 0, 10.0, None)
        
        children = agent.expandNode(rootNode)
        
        # Should generate some child nodes
        self.assertIsInstance(children, list)
        self.assertGreater(len(children), 0)
        
        # Each child should have the root as parent and depth 1
        for child in children:
            self.assertEqual(child.parent, rootNode)
            self.assertEqual(child.depth, 1)
            self.assertIsNotNone(child.move)
    
    def test_getMove_ReturnsValidMove(self):
        # Test that getMove returns a valid Move object
        myAnts = [
            Ant((0,0), QUEEN, 0), 
            Ant((1,0), WORKER, 0)
        ]
        enemyAnts = [Ant((0,9), QUEEN, 1)]
        
        anthill = Construction((0,0), ANTHILL)
        tunnel = Construction((1,0), TUNNEL)
        food = Construction((5,5), FOOD)
        
        myInv = Inventory(0, myAnts, [anthill, tunnel], 2)
        enemyInv = Inventory(1, enemyAnts, [Construction((0,9), ANTHILL)], 2)
        neutralInv = Inventory(2, [], [food], 0)
        
        board = [[Location((x,y)) for y in range(10)] for x in range(10)]
        state = GameState(board, [myInv, enemyInv, neutralInv], 0, 0)
        
        agent = AIPlayer(0)
        result = agent.getMove(state)
        
        # Should return a Move object
        self.assertIsInstance(result, Move)
        self.assertIsNotNone(result.moveType)
    
if __name__ == "__main__":
    unittest.main()