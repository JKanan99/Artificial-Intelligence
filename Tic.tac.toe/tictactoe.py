"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_cells, o_cells = 0, 0

    rows = len(board)
    cols = len(board[0])

    # Loop through each cell of the board
    # and count each X and O cell
    for row in range(rows):
        for col in range(cols):
            if board[row][col] == X:
                x_cells += 1
            if board[row][col] == O:
                o_cells += 1

    # O's turn if num of X cells > O cells
    # Otherwise, X's turn
    return O if x_cells > o_cells else X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()

    # Check if row, col position is empty
    # If so, add it to our possible moves set
    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col] == EMPTY:
                possible_actions.add((row, col))

    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    possible_actions = actions(board)
    if action not in possible_actions:
        # Invalid action was passed in
        raise Exception(f'{action} is not a valid action.')

    row, col = action

    # Create a deep copy of original board
    # to leave original unmodified
    copy_of_board = copy.deepcopy(board)

    # Check whose move it is,
    # assign to row, col cell of board
    copy_of_board[row][col] = player(board)

    return copy_of_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    rows = len(board)
    for row in range(rows):
        if board[row][0] == board[row][1] == board[row][2]:
            # Automatically, we know all are the same. So if one is EMPTY
            # all cells in the row are EMPTY. So this is why this works.
            if board[row][0] != EMPTY:
                return board[row][0]  # Returns X or O

    cols = len(board[0])
    for col in range(cols):
        if board[0][col] == board[1][col] == board[2][col]:
            # Automatically, we know all are the same. So if one is EMPTY
            # all cells in the column are EMPTY. So this is why this works.
            if board[0][col] != EMPTY:
                return board[0][col]  # Returns X or O

    # Once again, we know if there is a diagonal winner
    # then the diagonal will either be all X or all Y
    if board[0][0] == board[1][1] == board[2][2]:
        if board[0][0] != EMPTY:
            return board[0][0]  # Returns X or O

    if board[2][0] == board[1][1] == board[0][2]:
        if board[2][0] != EMPTY:
            return board[2][0]  # Returns X or O

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """

    # checking if there is a winner
    if winner(board) != None:
        return True

    rows = len(board)
    cols = len(board[0])

    # checking if there are any empty cells; if so, return false - game still in progress
    for row in range(rows):
        for col in range(cols):
            if board[row][col] == EMPTY:
                return False

    # if there are no more empty cells; theres a tie and game's over
    return True


def score(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """

    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def max_value(board):

    if terminal(board):
        return score(board)
    v = -math.inf

    for action in actions(board):
        v = max(v, min_value(result(board, action)))
    return v


def min_value(board):
    if terminal(board):
        return score(board)
    v = math.inf

    for action in actions(board):
        v = min(v, max_value(result(board, action)))
    return v


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    """
    Psuedocode as described by professor:
    
    Def minimax(board):
        Def max_function(state):
            score = max(score, min_function(results(state, actions)))
            Return score
            
        Def min_function(state):
            Return score
    """

    if terminal(board):
        return None

    best_action = None

    if player(board) == X:
        max_v = -math.inf
        for action in actions(board):
            v = min_value(result(board, action))
            if v > max_v:
                max_v = v
                best_action = action
    else:
        min_v = math.inf
        for action in actions(board):
            v = max_value(result(board, action))
            if v < min_v:
                min_v = v
                best_action = action

    return best_action