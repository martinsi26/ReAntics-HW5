import random
import sys
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *


##
#SearchNode
#Description: Represents a node in the search tree for AI decision making
#
#Variables:
#   move - The Move that would be taken to reach this node from parent
#   state - The GameState that would be reached by making the move
#   depth - How many moves it takes to reach this node from current state
#   evaluation - Sum of utility value and depth
#   parent - Reference to the parent node (for Part B)
##
class SearchNode:
    
    def __init__(self, move, state, depth, parent=None):
        self.move = move
        self.state = state
        self.depth = depth
        self.parent = parent
        self.evaluation = None  # Will be calculated when needed
    
    def calculate_evaluation(self, utility_function):
      
        #Calculate the evaluation value for this node.
        #Evaluation = utility(state) + depth
        
        utility_value = utility_function(self.state)
        self.evaluation = utility_value + self.depth
        return self.evaluation
    
    def to_dict(self):
        
        #Convert node to dictionary representation for easy access
        
        return {
            'move': self.move,
            'state': self.state,
            'depth': self.depth,
            'evaluation': self.evaluation,
            'parent': self.parent
        }
    
    def __str__(self):
        return f"SearchNode(move={self.move}, depth={self.depth}, eval={self.evaluation})"


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
        super(AIPlayer,self).__init__(inputPlayerId, "Siggy_HW5_rhoades26_martinsi26")
    
    def utility(self, gameState):
        # Simple utility function - evaluates game state from 0 to 1
        # 0.5 = neutral, close to 1.0 = winning, close to 0 = losing
        
        # Use the current player's turn to determine which inventory to use
        my_inventory = gameState.inventories[gameState.whoseTurn]
        enemy_inventory = gameState.inventories[1 - gameState.whoseTurn]
        anthill = my_inventory.getAnthill()

        foodPoint = getConstrList(gameState, gameState.whoseTurn, (TUNNEL,))[0]
        # Start with neutral score
        score = 0.5

        # Scaling Variables
        foodFactor = 0.03 # Points per food difference
        antFactor = 0.02
        attackerFactor = 0.05 # Importance of attacker difference
        threatFactor = 0.05
        baseEstimate = 30
        grabBonus = 0.006 # Bonus for picking up food and for putting it down
        
        # Food advantage 
        my_food = my_inventory.foodCount
        enemy_food = enemy_inventory.foodCount

        
        if my_food >= 11:  # FOOD_GOAL
            score = 0.9  # Almost winning
        else:
            # Simple food difference calculation
            food_diff = my_food - enemy_food
            score += food_diff * foodFactor  # Each food difference = 0.05 points
        
        # Queen status 
        my_queen = my_inventory.getQueen()
        enemy_queen = enemy_inventory.getQueen()
        
        if my_queen is None and enemy_queen is not None:
            score = 0  # My queen dead, enemy alive = losing badly
        elif my_queen is not None and enemy_queen is None:
            score = 1  # Enemy queen dead, I'm alive = winning
        
        # Ant count advantage
        my_ant_count = len(my_inventory.ants)
        enemy_ant_count = len(enemy_inventory.ants)
        my_attacker_count = len(list(filter(lambda ant: ant.type in [2, 3, 4], my_inventory.ants)))
        enemy_attacker_count = len(list(filter(lambda ant: ant.type in [2, 3, 4], enemy_inventory.ants)))
        
        if my_ant_count + enemy_ant_count > 0:
            ant_ratio = my_ant_count / (my_ant_count + enemy_ant_count)
            #score += (ant_ratio - 0.5) * 0.2  # Ant advantage contributes up to 0.1
            score += ant_ratio * antFactor * (1 - attackerFactor)
        
        if my_attacker_count + enemy_attacker_count > 0:
            ant_ratio = my_attacker_count / (my_attacker_count + enemy_attacker_count)
            score += ant_ratio * antFactor * attackerFactor
        
        # Written by Claude
        if anthill is not None:
            enemy_attackers = [ant for ant in enemy_inventory.ants if ant.type in [2, 3, 4]]
            for attacker in enemy_attackers:
                threat_distance = stepsToReach(gameState, attacker.coords, anthill.coords)
                if threat_distance <= 3:
                    score -= threatFactor * (4 - threat_distance) / 10
                elif threat_distance <= 6:
                    score -= threatFactor * (7 - threat_distance) / 40
        
        # food distance scoring
        distances = []
        progressScores = []

        tunnel = foodPoint  # This is the tunnel
        
        for ant in filter(lambda ant: ant.type == 1, my_inventory.ants): # For each worker
            if ant.carrying:
                # Check distance to both anthill and tunnel for drop-off
                dist_to_anthill = stepsToReach(gameState, ant.coords, anthill.coords)
                dist_to_tunnel = stepsToReach(gameState, ant.coords, tunnel.coords)
                min_dropoff_dist = min(dist_to_anthill, dist_to_tunnel)
                dist = min_dropoff_dist  # Set dist for the carrying case
                
                if min_dropoff_dist == 0:
                    score += grabBonus  # Very strong reward for being at drop-off point with food
                else:
                    # Reward progress toward dropping off food
                    progress = max(0, (1 - min_dropoff_dist / 12) * 0.0003) # progress based on distance to drop-off
                    score += progress
            else:
                foods = getConstrList(gameState, None, (FOOD,)) #this section is just the foodgatherer distance code but modified to work here
                bestDistSoFar = 1000 #i.e., infinity
                for food in foods:

                    foodSafe = True

                    for loc in listAdjacent(food.coords):
                        if not getAntAt(gameState, food.coords) == None:
                            foodSafe = False

                    if getAntAt(gameState, food.coords) == None and foodSafe == True:
                        d = stepsToReach(gameState, ant.coords, food.coords)
                        if (d < bestDistSoFar):
                            bestDistSoFar = d
                dist = bestDistSoFar

                progress = max(0, foodFactor * (1 - dist / 12))
                progressScores.append(progress)
                if dist == 0:
                    score += grabBonus
            distances.append(dist)

        if progressScores:
            foodScore = sum(progressScores) / len(progressScores) # gets an average score
            #print(foodScore)
            score += 0.0001 * foodScore #scales score based on how important we want this to be to our algorithm.
        
        # Keep score between 0 and 1

        score = max(0.0, min(1.0, score))

        inverseScore = 1 - score
        movesLeft = baseEstimate * 2 * inverseScore
        
        return movesLeft
    
    def create_search_node(self, move, state, depth=1, parent=None):
        #
        #Helper method to create a SearchNode with the given parameters.
        #Automatically calculates evaluation using this player's utility function.
        
        node = SearchNode(move, state, depth, parent)
        node.calculate_evaluation(self.utility)
        return node
    
    def bestMove(self, nodes):
        #
        #Helper method to find the node with the best evaluation from a list of SearchNodes.
        #Returns the SearchNode with the highest evaluation value.
        #If the list is empty, returns None.
        #
        #Parameters:
        #   nodes - List of SearchNode objects to evaluate
        #
        #Return: SearchNode with the best evaluation, or None if list is empty
        
        if not nodes:
            return None
        
        # Find the node with the maximum evaluation
        best_node = max(nodes, key=lambda node: node.evaluation)
        return best_node
    
    def expandNode(self, node):
        #
        #Expands a given node by generating all valid moves from its GameState
        #and creating properly initialized child nodes for each move.
        #
        #Parameters:
        #   node - The SearchNode to expand (contains the GameState to generate moves from)
        #
        #Return: List of new SearchNode objects representing all valid moves from the given node
        
        # Get all legal moves from the current state
        moves = listAllLegalMoves(node.state)
        
        # Filter out build moves if we already have 3+ ants
        numAnts = len(node.state.inventories[node.state.whoseTurn].ants)
        if numAnts >= 3:
            moves = [move for move in moves if move.moveType != BUILD]
        
        # Create search nodes for each legal move
        child_nodes = []
        for move in moves:
            # Create a new state by applying the move
            new_state = getNextState(node.state, move)
            
            # Create search node with this move and new state
            # Depth is parent's depth + 1, parent is the given node
            child_node = self.create_search_node(move, new_state, depth=node.depth + 1, parent=node)
            child_nodes.append(child_node)
        
        return child_nodes
    

    
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
        numToPlace = 0
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
    
    ##
    #getMove
    #Description: Gets the next move from the Player.
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: The Move to be made
    ##
    def getMove(self, currentState):
        # a. Create two empty lists
        frontierNodes = []  # nodes that have not yet been expanded
        expandedNodes = []  # nodes that have been expanded

        maxDepth = 3
        depth = 0
        
        # b. Create a root node from the current state
        # Root node has no move, depth 0, and no parent
        root_node = SearchNode(None, currentState, 0, None)
        root_node.calculate_evaluation(self.utility)  # Calculate evaluation for root
        frontierNodes.append(root_node)
        count = 0
        # c. Perform the search loop
        while frontierNodes and count < 3:
            count += 1
            # Select the node with the best (lowest) score from frontierNodes
            # Note: "best" means lowest score in this context
            best_node = min(frontierNodes, key=lambda node: node.evaluation)
            
            # Remove it from frontierNodes and add to expandedNodes
            frontierNodes.remove(best_node)
            expandedNodes.append(best_node)
            
            # Call expandNode on the selected node
            new_nodes = self.expandNode(best_node)
            
            # Add all new nodes to frontierNodes
            frontierNodes.extend(new_nodes)
        
        # d. Find the node with the best (lowest) score in frontierNodes
        if not frontierNodes:
            # If no frontier nodes, fallback to random move
            moves = listAllLegalMoves(currentState)
            numAnts = len(currentState.inventories[currentState.whoseTurn].ants)
            if numAnts >= 3:
                moves = [move for move in moves if move.moveType != BUILD]
            return moves[random.randint(0, len(moves) - 1)] if moves else None
        
        best_frontier_node = min(frontierNodes, key=lambda node: node.evaluation)
        
        # Trace back to the initial move (depth 1)
        current_node = best_frontier_node
        while current_node.parent is not None and current_node.parent.parent is not None:
            current_node = current_node.parent
        
        # Return the move that led to this best node
        if random.randint(1, 30) == 1:
            return random.choice(listAllLegalMoves(currentState))
        else:
            return current_node.move

        return current_node.move
    
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
