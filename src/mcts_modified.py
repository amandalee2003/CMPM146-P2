
from mcts_node import MCTSNode
from p2_t3 import Board
from random import choice
from math import sqrt, log

num_nodes = 1000
explore_faction = 2.

# Selection Stage
# The goal here is to traverse down the most interesting/optimal path until we reach a leaf to expand
def traverse_nodes(node: MCTSNode, board: Board): #, bot_identity: int):
    """ Traverses the tree until the end criterion are met.
    e.g. find the best expandable node (node with untried action) if it exist,
    or else a terminal node

    Args:
        node:       A tree node from which the search is traversing.
        board:      The game setup.
        identity:   The bot's identity, either 1 or 2

    Returns:
        node: A node from which the next stage of the search can proceed.

    """
    # is_opponent = board.current_player(state) != bot_identity
    
    tmp = node
    while not board.is_ended(tmp.state):
        if len(tmp.untried_actions) != 0:
            return expand_leaf(tmp, board)
        else:
            best_node = None

            # find node with best UCB
            for _, n in tmp.child_nodes.items():
                if best_node == None or ucb(n) > ucb(best_node):
                    best_node = n

            tmp = best_node

    return tmp

# Expansion Stage
# Here we generate random legal moves and pick one to simulate
# For vanilla MCTS, I think this just means completely random moves,
# but for modified MCTS we can use a heuristic to estimate optimal
# moves
def expand_leaf(node: MCTSNode, board: Board):
    """ Adds a new leaf to the tree by creating a new child node for the given node (if it is non-terminal).

    Args:
        node:   The node for which a child will be added.
        board:  The game setup.

    Returns:
        node: The added child node

    """
    action = node.untried_actions.pop()
    new_state = board.next_state(node.state, action)
    new_node = MCTSNode(new_state, parent=node, parent_action=action, action_list=board.legal_actions(new_state))
    node.child_nodes[action] = new_node

    return new_node

import heapq

def action_score(player, pieces, action):
    ay, ax = action[0], action[1]
    ap = (action[2], action[3])
    
    row_p, row_o, row_e = 0, 0, 0
    col_p, col_o, col_e = 0, 0, 0
    dl_p, dl_o, dl_e = 0, 0, 0
    dr_p, dr_o, dr_e = 0, 0, 0

    # scan row
    for x in range(0, 3):
        space = pieces[(ay, ax, ap[0], x)] if (ay, ax, ap[0], x) in pieces else 0
        if space == player:
            row_p += 1
        elif space != 0:
            row_o += 1
            row_p = -2
        else:
            row_e += 1

    if row_p < 0:
        row_p = 0

    # scan column
    for y in range(0, 3):
        space = pieces[(ay, ax, y, ap[1])] if (ay, ax, y, ap[1]) in pieces else 0
        if space == player:
            col_p += 1
        elif space != 0:
            col_o += 1
            col_p = -2
        else:
            col_e += 1

    if col_p < 0:
        col_p = 0

    # if action is diagonal left
    if ap == (0, 0) or ap == (1, 1) or ap == (2, 2):
        dl1 = pieces[(ay, ax, 0, 0)] if (ay, ax, 0, 0) in pieces else 0
        dl2 = pieces[(ay, ax, 1, 1)] if (ay, ax, 1, 1) in pieces else 0
        dl3 = pieces[(ay, ax, 2, 2)] if (ay, ax, 2, 2) in pieces else 0
        
        if dl1 == player:
            dl_p += 1
        elif dl1 != 0:
            dl_o += 1
            dl_p = -2
        else:
            dl_e += 1

        if dl2 == player:
            dl_p += 1
        elif dl2 != 0:
            dl_o += 1
            dl_p = -2
        else:
            dl_e += 1

        if dl3 == player:
            dl_p += 1
        elif dl3 != 0:
            dl_o += 1
            dl_p = -2
        else:
            dl_e += 1

        if dl_p < 0:
            dl_p = 0
    
    # if action is diagonal right
    if ap == (2, 0) or ap == (1, 1) or ap == (0, 2):
        dr1 = pieces[(ay, ax, 2, 0)] if (ay, ax, 2, 0) in pieces else 0
        dr2 = pieces[(ay, ax, 1, 1)] if (ay, ax, 1, 1) in pieces else 0
        dr3 = pieces[(ay, ax, 0, 2)] if (ay, ax, 0, 2) in pieces else 0
        
        if dr1 == player:
            dr_p += 1
        elif dr1 != 0:
            dr_o += 1
            dr_p = -2
        else:
            dr_e += 1

        if dr2 == player:
            dr_p += 1
        elif dr2 != 0:
            dr_o += 1
            dr_p = -2
        else:
            dr_e += 1

        if dr3 == player:
            dr_p += 1
        elif dr3 != 0:
            dr_o += 1
            dr_p = -2
        else:
            dr_e += 1

        if dr_p < 0:
            dr_p = 0

    # compute final score
    pm = 3
    om = 2
    em = 0.3
    
    score = (pow(row_p, 2) * pm + pow(row_o, 2) * om + pow(col_p, 2) * pm + pow(col_o, 2) * om + pow(dl_p, 2) * pm + pow(dl_o, 2) * om + dl_e * em + pow(dr_p, 2) * pm + pow(dr_o, 2) * om + (row_e + col_e + dr_e) * em)
    
    # print(action, "Row: " + str((row_p, row_o, row_e)), "Col: " + str((col_p, col_o, col_e)), "DL: " + str((dl_p, dl_o, dl_e)), "DR: " + str((dr_p, dr_o, dr_e)), "Score: " + str(score))
    
    return 0 - score

def rollout_policy(board: Board, state):
    current_player = board.current_player(state)
    actions = board.legal_actions(state)
    
    unpacked_state = board.unpack_state(state)
    pieces = {}

    # unpack piece placements in current box
    for piece in unpacked_state['pieces']:
        oy, ox = piece['outer-row'], piece['outer-column']
        iy, ix = piece['inner-row'], piece['inner-column']

        pieces[(oy, ox, iy, ix)] = piece['player']

    priority_queue = []

    # for each legal action, determine priority
    for action in actions:
        score = action_score(current_player, pieces, action)

        # another heuristic to favor forcing opponent into another board
        if action[0] == action[2] and action[1] == action[3]:
            score += 5

        # yet another heuristic to favor picking spaces that dont give opponent freebie
        if board.owned_boxes(state)[(action[2], action[3])] != 0:
            score += 3

        # print('Final score', score)
        
        heapq.heappush(priority_queue, (score, action))

    priority, action = heapq.heappop(priority_queue)
    
    # print(priority, action)
    
    return action

# Simulation Stage
# Here we randomly chose legal moves for both opponents until we reach an end state (win, lose, draw)
def rollout(board: Board, state):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        board:  The game setup.
        state:  The state of the game.
    
    Returns:
        state: The terminal game state

    """
    tmp_state = state
    while not board.is_ended(tmp_state):
        # action = choice(board.legal_actions(tmp_state))
        # print(board.display(tmp_state, None))
        action = rollout_policy(board, tmp_state)
        tmp_state = board.next_state(tmp_state, action)

    return tmp_state

# Backpropagation Stage
# This is just walking back up the tree to the root, incrementing the win/play values of each node as it passes by them
# Im not sure if we should consider a draw as a loss or not
def backpropagate(node: MCTSNode|None, won: bool):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        won:    An indicator of whether the bot won or lost the game.

    """
    if node == None:
        return

    tmp = node
    while tmp != None:
        if won:
            tmp.wins += 1
        tmp.visits += 1
        tmp = tmp.parent

# Upper Confidence Bound (https://en.wikipedia.org/wiki/Thompson_sampling#Upper-Confidence-Bound_(UCB)_Algorithms)
# Formula for MCTS: w / n + c * sqrt(ln(p) / n)
# where w = total wins for current node, n = visit count for current node, p = visit count for parent node, c = constant that controls the rate of exploration (usually set to sqrt(2))
# w / n = the average win rate of the node, which makes the algorithm favor well performing paths
# c * sqrt(ln(p) / n) = represents urge to explore, gets smaller as node is explored more, so pushes the algorithm to explore less explored nodes
def ucb(node: MCTSNode, explore_rate=sqrt(2)): #, is_opponent: bool):
    """ Calcualtes the UCB value for the given node from the perspective of the bot

    Args:
        node:   A node.
        explore_rate: influence the exploration rate
    Returns:
        The value of the UCB function for the given node
    """
    average_win_rate = 0
    explore = 0

    if node.visits != 0:
        average_win_rate = node.wins / node.visits
        explore = sqrt(2 * log(node.parent.visits) / node.visits)

    return average_win_rate + explore_rate * explore

# This should pick which child node has the best likely outcome/path
def get_best_action(root_node: MCTSNode):
    """ Selects the best action from the root node in the MCTS tree

    Args:
        root_node:   The root node
    Returns:
        action: The best action from the root node
    
    """
    if len(root_node.child_nodes) == 0:
        return None # something went horribly wrong if this is reached
    
    best_action = None

    # find action with best win rate
    for action, node in root_node.child_nodes.items():
        if best_action == None or ucb(node, 0.0) > ucb(root_node.child_nodes[best_action], 0.0):
            best_action = action
    
    return best_action

def is_win(board: Board, state, identity_of_bot: int):
    # checks if state is a win state for identity_of_bot
    outcome = board.points_values(state)
    assert outcome is not None, "is_win was called on a non-terminal state"
    return outcome[identity_of_bot] == 1

def think(board: Board, current_state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        current_state:  The current state of the game.

    Returns:    The action to be taken from the current state

    """
    bot_identity = board.current_player(current_state) # 1 or 2
    root_node = MCTSNode(current_state, parent=None, parent_action=None, action_list=board.legal_actions(current_state))

    for _ in range(num_nodes):
        # print("Selection Phase")
        node = traverse_nodes(root_node, board) #, bot_identity)

        if node == None:
            print("something went wrong")
            break

        # print("Simulation Phase")
        end_state = rollout(board, node.state)

        # print("Backpropagation Phase")
        backpropagate(node, is_win(board, end_state, bot_identity))

    # print(root_node.tree_to_string())

    # Return an action, typically the most frequently used action (from the root) or the action with the best
    # estimated win rate.
    best_action = get_best_action(root_node)
    
    print(f"Action chosen: {best_action}")
    return best_action
