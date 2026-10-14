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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        # αριχκοποιώ το σκορ
        score = 0

        # υπολογίζω απόσταση από εγγύτερη τροφή με την μανχάταν
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()] # asList όπως μας λέει και η εκφώνηση
        if foodDistances:
            score += 10.0 / min(foodDistances)  # πρώτα το κοντινότερο φαγητό

        # κώδικας για φαντάσματα - ζιπ (λίστα, λίστα)
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghostState.getPosition()
            ghostDistance = manhattanDistance(newPos, ghostPos)
        
            if scaredTime == 0:  # Αν το φάντασμα δεν είναι φοβισμένο, με χρώμα δηλαδή
                if ghostDistance < 2:  # Κίνδυνος να φαγωθεί
                    score -= 100  # Αποφυγή φαντάσματος
            else:
                # Αν το φάντασμα είναι φοβισμένο, θέλουμε να το φάμε
                score += 20.0 / ghostDistance

        # για να μην σταματάει χωρίς λόγο
        if action == Directions.STOP:
            score -= 10

        return successorGameState.getScore() + score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        def minimax(state, agentIndex, currentDepth):
            # κάνω έλεγχο για τερματισμό (νίκη, ήττα ή μέγιστο βαθος)
            if state.isWin() or state.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(state)
            
            # παίζει ο Pacman (max)
            if agentIndex == 0:
                return max_value(state, currentDepth)
            # παίζουν τα φαντάσματα (min)
            else:
                return min_value(state, agentIndex, currentDepth)
        
        def max_value(state, currentDepth):
            # αρχικοποιώ για το καλύτερο σκόρ και την καλύτερη κίνηση
            best_score = float('-inf')
            best_action = None
            # για κάθε έγκυρη ενέργεια του Pacman:
            for action in state.getLegalActions(0):
                # δημιουργείται παιδί μετά την κίνηση
                successor = state.generateSuccessor(0, action)
                # καλείται η minimax για τα φαντάσματα
                score = minimax(successor, 1, currentDepth)  
                
                # αν βρώ καλύτερη τιμή σκορ, κρατάω και τα 2
                if score > best_score:
                    best_score = score
                    best_action = action
                    
            # στο root επιστρέφω το καλύτερο σκορ
            if currentDepth == 0:
                return best_action  
            else:
                return best_score  
        
        def min_value(state, agentIndex, currentDepth):
            # αρχικοποιώ την min τιμή του σκορ
            min_score = float('inf')
            # για κάθε έγκυρη ενέργεια των φαντασμάτων:
            for action in state.getLegalActions(agentIndex):
                # δημιουργείται παιδί μετά την κίνηση
                successor = state.generateSuccessor(agentIndex, action)
                
                # ελέγχω αν είναι το τελευταίο φάντασμα
                if agentIndex == state.getNumAgents() - 1:
                    # αν ναι, παίζει ο Pacman
                    score = minimax(successor, 0, currentDepth + 1)  # Back to Pacman at next depth
                else:
                    # αν όχι, παίζει το επόμενο φάντασμα
                    score = minimax(successor, agentIndex + 1, currentDepth)  # Next ghost
                #αν βρούμε min, το κρατάμε 
                min_score = min(min_score, score)
            
            return min_score
        
        # αρχίζουμε το minimax από τη ρίζα με τον Pacman (agent index 0) και βάθος 0
        return minimax(gameState, 0, 0)
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alpha_beta(state, agentIndex, currentDepth, alpha, beta):
            # κάνω έλεγχο για τερματισμό (νίκη, ήττα ή μέγιστο βαθος)
            if state.isWin() or state.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(state)
            
            # παίζει ο Pacman (max)
            if agentIndex == 0:
                return max_value(state, currentDepth, alpha, beta)
            # παίζουν τα φαντάσματα (min)
            else:
                return min_value(state, agentIndex, currentDepth, alpha, beta)
        
        def max_value(state, currentDepth, alpha, beta):
            # αρχικοποιώ το καλύτερο σκορ ως -00 και την καλύτερη κίνηση ως καμία
            best_score = float('-inf')
            best_action = None
            # για κάθε νόμιμη κίνηση του Pacman
            for action in state.getLegalActions(0):
                # μεταβαίνω στο παιδί 
                successor = state.generateSuccessor(0, action)
                # καλείται η συνάρτηση για τα φαντάσματα
                score = alpha_beta(successor, 1, currentDepth, alpha, beta)
                
                # αν βρώ καλύτερο σκορ, κρατάω σκορ και κίνηση
                if score > best_score:
                    best_score = score
                    best_action = action
                
                # ενημερώνω την τιμή του άλφα με το μέγιστο εως τώρα
                alpha = max(alpha, score)
                
                # κλαδέυω (pruning) αν ισχύει η συνθήκη (το = δεν το θέλουμε)
                if beta < alpha:            # γιατί δεν θα εξερευνήσει το πλήθος κόμβων που θέλουμε
                    break
            
            # αν είμαστε στην ρίζα επιστρέφουμε την καλύτερη κίνηση, αλλιώς επιστρέφουμε την καλύτερη τιμή
            if currentDepth == 0:
                return best_action
            else:
                return best_score
        
        def min_value(state, agentIndex, currentDepth, alpha, beta):
            # αρχικοποιώ το ελάχιστο σκορ σε +00
            min_score = float('inf')
            # για κάθε νόμιμη ενέργεια των φαντασμάτων
            for action in state.getLegalActions(agentIndex):
                # μεταβαίνω στο παιδί 
                successor = state.generateSuccessor(agentIndex, action)
                
                # σε περίπτωση είναι το τελευταίο φάντασμα, συνεχίζουμε με τον Pacman στο επόμενο βάθος
                if agentIndex == state.getNumAgents() - 1:
                    score = alpha_beta(successor, 0, currentDepth + 1, alpha, beta)
                # ειδάλως συνεχίζουμε με το επόμενο φάντασμα
                else:
                    score = alpha_beta(successor, agentIndex + 1, currentDepth, alpha, beta)
                
                # κρατάω τηω ελάχιστη τιμή αν βρώ μικρότερη
                min_score = min(min_score, score)
                
                # ενημερώνω την τιμή του βήτα με το ελάχιστο έως τώρα
                beta = min(beta, score)
                
                # κλαδέυω (pruning) αν ισχύει η συνθήκη (το = δεν το θέλουμε)
                if beta < alpha:        # γιατί δεν θα εξερευνήσει το πλήθος κόμβων που θέλουμε
                    break
            
            return min_score

        # αρχίζουμε το alpha-beta από τη ρίζα με τον Pacman (agent index 0), βάθος 0, και αρχικές τιμές για άλφα και βήτα
        return alpha_beta(gameState, 0, 0, float('-inf'), float('inf'))

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(state, agentIndex, currentDepth):
            # κάνω έλεγχο για τερματισμό: νίκη, ήττα ή μέγιστο βάθος
            if state.isWin() or state.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(state)
            
            # παίζει ο Pacman (max)
            if agentIndex == 0:
                return max_value(state, currentDepth)
            # παίζουν τα φαντάσματα
            else:
                return expect_value(state, agentIndex, currentDepth)

        def max_value(state, currentDepth):
            #αρχικοποιώ το καλύτερο σκορ στο -00 και την καλύτερη κίνηση ως καμία
            best_score = float('-inf')
            best_action = None
            # για κάθε νόμιμη πράξη του Pacman
            for action in state.getLegalActions(0):
                # πήγαινε στο παιδί
                successor = state.generateSuccessor(0, action)
                # κάλεσε την συνάρτηση για τα φαντάσματα
                score = expectimax(successor, 1, currentDepth)
                
                # κρατάμε το καλύτερο σκορ και κίνηση, αν βρούμε το καλύτερο σκορ
                if score > best_score:
                    best_score = score
                    best_action = action

            # επιστρέφουμε την καλύτερη κίνηση, αλλιώς το καλύτερο σκορ
            if currentDepth == 0:
                return best_action
            else:
                return best_score

        def expect_value(state, agentIndex, currentDepth):
            # νόμιμες κινήσεις φαντασμάτων ως κόμβοι τύχης
            actions = state.getLegalActions(agentIndex)
            num_actions = len(actions)
            if num_actions == 0: # ελέγχω αν κινούνται
                return 0
            
            expected_score = 0
            # Υπολογίζω το αναμενόμενο σκορ ως το μέσο όρο των σκορ για τις πιθανές κινήσεις
            for action in actions:
                # πήγαινε στο επόμενο παιδί
                successor = state.generateSuccessor(agentIndex, action)
                
                if agentIndex == state.getNumAgents() - 1:  # αν φτάσαμε στο τελευταίο φάντασμα, πάμε στον Pacman
                    score = expectimax(successor, 0, currentDepth + 1)
                else:  # επόμενο φάντασμα
                    score = expectimax(successor, agentIndex + 1, currentDepth)
                #ενημερώνω την τιμή του αναμενόμενου σκορ
                expected_score += score / num_actions

            return expected_score

        # Ξεκινάμε το expectimax από τη ρίζα με τον Pacman (agent index 0) και βάθος 0 όπως στο minimax
        return expectimax(gameState, 0, 0)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    # ανοίγω το pacman.py και από το Gamestate παίρνω τα παρακάτω:
    pacman_pos = currentGameState.getPacmanPosition()   # θεσή Pacman
    food = currentGameState.getFood()                   # κουκίδες φαγητού
    ghost_states = currentGameState.getGhostStates()    # κατάσταση φαντασμάτων
    capsules = currentGameState.getCapsules()           # κάψουλες ή power ups ή energy pellet όπως τα λένε (αυτά που κάνουν vulnerable τα φαντάσματα)
    score = currentGameState.getScore()                 # και το σκορ

    
    # υπολογίζω την κοντινότερη απόσταση φαγητού
    food_distances = [manhattanDistance(pacman_pos, food_pos) for food_pos in food.asList()]
    if food_distances: # λίστα όπως δημιουργήθηκε παραπάνω, αν περιέχει τιμές
        closest_food_distance = min(food_distances)
    else:
        closest_food_distance = 0  # Δεν υπάρχει φαγητό, οπότε δεν επηρεάζει
    
    # υπολογίζω απόσταση από φαντάσματα
    ghost_distances = []
    for ghost in ghost_states:
        ghost_distance = manhattanDistance(pacman_pos, ghost.getPosition())
        if ghost.scaredTimer == 0:  # το φάντασμα δεν είναι vulnerable
            ghost_distances.append(ghost_distance) # προσθέτει την απόσταση στην λίστα
        elif ghost_distance <= 2:  # αν είναι vulnerable και κοντά, το κυνηγάμε
            ghost_distances.append(-2)  # αρνητικό γιατί ετσί θα στραφεί προς το vulnerable φάντασμα
        
    # καθορίζω την πλησιέστερη απόσταση (αν υπάρχει φυσικά)
    closest_ghost_distance = min(ghost_distances) if ghost_distances else float('inf')

    # υπολογίζω απόσταση από κάψουλες ή power ups ή energy pellets
    capsule_distances = [manhattanDistance(pacman_pos, capsule) for capsule in capsules]
    if capsule_distances: # λίστα όπως δημιουργήθηκε παραπάνω, αν περιέχει τιμές
        closest_capsule_distance = min(capsule_distances)
    else:
        closest_capsule_distance = 0  # Αν δεν υπάρχουν κάψουλες, δεν επηρεάζει

    # εκτιμώ τον αλγόριθμο βάσει των παρακάτω 3 παραγόντων, ήτοι:
    evaluation = score # βάζω το σκορ
    if closest_food_distance > 0: # αντιστρέφω απόσταση για φαγητό - αν χ>1 πιο επιθετικός ο Pacman
        evaluation += 1 / closest_food_distance # έγιναν δοκιμές με 1,3,10,20 
    if closest_ghost_distance > 0: # αντιστρέφω απόσταση για φαντάσματα - αν χ>5 τα πλησιάζει περισσότερο
        evaluation -= 5 / closest_ghost_distance # έγιναν δοκιμές με 1,3,5 
    if closest_capsule_distance > 0: #αντιστρεφω απόσταση από κάψουλες - αν χ>5 τις πλησιάζει περισσότερο
        evaluation += 5 / closest_capsule_distance # έγιναν δοκιμές με 1,3,5 
    evaluation -= 3 * len(food.asList()) # αν χ>2 μαζέυει πιο γρήγορα το φαγητό
    # έγιναν δοκιμές με 1,2,3 
    return evaluation

# Abbreviation
better = betterEvaluationFunction
