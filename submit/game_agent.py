"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Calculates a quick heuristic, inaccurate but allows for deep searches

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # check if the game is won or lost
    if game.is_loser(player): return float("-inf")
    if game.is_winner(player): return float("inf")

    # calculate a simple heurisitic that is quick to compute and allows for deep searches during early game
    # start with the simplest heuristic, #moves
    player_moves =  len(game.get_legal_moves(player)) / (game.width * game.height)

    # calculate #opponent moves
    opp_moves = len(game.get_legal_moves(game.get_opponent(player))) / (game.width * game.height)

    # but a stronger emphasis on restricting opponents movement
    return player_moves - 2 * opp_moves


    # try to stay away from the opponent
    # heuristic = sum([abs(a-b) for a,b in zip(game.get_player_location(player), game.get_player_location(game.get_opponent(player)))]) / (game.width * game.height)

    # penalise borders of the board
    # heuristic = sum([abs(a-b) for a,b in zip(game.get_player_location(player), ((game.height-1)/2, (game.width-1)/2))]) / (game.width * game.height)



def custom_score_late(game, player, startgame, time_left, TIMER_THRESHOLD):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    A more cost-intensive heuristic, limits search depth but useful during later game stages.

    Note: this function should be called from within a Player instance as
    `self.score_late()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # check if the game is won or lost
    if game.is_loser(player): return float("-inf")
    if game.is_winner(player): return float("inf")

    # simple check for divided board
    # if the board is divided horizontally or vertically by a line of width>=2,
    # and the opponents are in different halfs, the player in the bigger half wins
    blanks = game.get_blank_spaces()
    playerpos = game.get_player_location(player)
    opppos = game.get_player_location(game.get_opponent(player))

    # check if two adjacent rows or columns are completely blocked off
    rows, cols = zip(*blanks)
    rowcount = [rows.count(x) for x in range(game.height)]
    colscount = [cols.count(x) for x in range(game.width)]
    try:
        hwall = [sum(x) for x in zip(rowcount[:-1], rowcount[1:])].index(0)
    except:
        hwall = False
    try:
        vwall = [sum(x) for x in zip(colscount[:-1], colscount[1:])].index(0)
    except:
        vwall = False

    # if there is a wall, check if players are on opposite sides of it (not on it)
    if (hwall and ((playerpos[0] < hwall < opppos[0]-1) or (opppos[0] < hwall < playerpos[0]-1))) or (vwall and ((playerpos[1] < vwall < opppos[1]-1) or (opppos[1] < vwall < playerpos[1]-1))):

        # players are on separate islands and can not interfere with each other,
        # find out which player has the longest chain of moves available
        try:
            player_chain = get_chain_length(game, player, time_left, TIMER_THRESHOLD)
            opp_chain = get_chain_length(game, game.get_opponent(player), time_left, TIMER_THRESHOLD)
        except Timeout:
            raise Timeout()

        return player_chain - opp_chain

    # return the early game heuristic if board cannot be divided
    return custom_score(game, player)


def get_chain_length(game, player, time_left, TIMER_THRESHOLD):
    """Calculate the longest chain of moves. If the players are separated, the player with more available moves wins.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    max_moves = 0
    if time_left() < TIMER_THRESHOLD:
        raise Timeout()

    legal_moves = game.get_legal_moves(player)
    if not legal_moves: return 1

    for move in legal_moves:
        nmoves = 1
        new_game = game.forecast_move(move)
        new_game.__active_player__ = player
        nmoves += get_chain_length(new_game, player, time_left, TIMER_THRESHOLD)
        if nmoves > max_moves: max_moves = nmoves
    return max_moves


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        if self.score == custom_score: self.score_late = custom_score_late
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.legal_moves = []
        self.nextmove = (-1, -1)
        self.nextscore = 0
        self.currentgame = []
        self.depth = 0
        self.depthlist = []


    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left
        self.legal_moves = legal_moves
        self.currentgame = game

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        best_move = (-1, -1)

        # check if any legal moves are available
        if not self.legal_moves: return best_move
        assert len(self.legal_moves) >= 0, 'Negative number of legal moves encountered!?'

        # pick an opening book
        best_move = self.find_opening(game)
        if best_move != (-1, -1): return best_move

        # assign a move in case non-iterative minimax or alphabeta times out
        best_move = legal_moves[0]

        try:
            if self.iterative:
                best_score, best_move = self.iterative_deepening(game, self.method)
            else:
                if self.method == 'minimax':
                    best_score, best_move = self.minimax(game, self.search_depth)
                elif self.method == 'alphabeta':
                    best_score, best_move = self.alphabeta(game, self.search_depth)
                else:
                    raise NotImplementedError
        except Timeout:
            best_move = self.nextmove
            best_score = self.nextscore

        # if it looks like the player loses, assign any move that's available
        if best_score == float("-inf"): best_move = legal_moves[0]

        # keep track of iterative depth reached
        if self.iterative: self.depthlist.append(self.depth)

        # Return the best move from the last completed search iteration
        return best_move


    def find_opening(self, game):
        """Decide how to open the game.
        This is not really used at the moment, placeholder in case more opening strategies are used in the future.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        Returns
        -------
        tuple(int, int)
            The best move for the beginning of the game; (-1, -1) if no good initial move is found
        """

        # if the board is empty
        if len(game.get_blank_spaces()) == game.height * game.width:
            # if there is a center, play there
            if game.height % 2 != 0 and game.width%2 != 0: return ((game.height+1)/2, (game.width+1)/2)

        return (-1, -1)


    def iterative_deepening(self, game, method='minimax'):
        """Implement the iterative deepening search.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        method : {'minimax', 'alphabeta'} (optional)
            The name of the search method to use in get_move().

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """

        self.depth = 1
        current_depth = 1
        self.nextscore = 0

        # increase the max search depth and run a search until we find a winning path
        while self.nextscore != float("inf") and self.nextscore != float("-inf"):
            try:
                if self.method == 'minimax':
                    score, move = self.minimax(game, current_depth)
                elif self.method == 'alphabeta':
                    score, move = self.alphabeta(game, current_depth)
                else:
                    raise NotImplementedError
                current_depth += 1
                self.nextmove = move
                self.nextscore = score
            except Timeout:
                self.depth = current_depth
                raise Timeout()
        self.depth = current_depth
        return self.nextscore, self.nextmove


    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """


        if not self.currentgame: self.currentgame = game

        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        legal_moves = game.get_legal_moves()
        assert len(legal_moves) >= 0, 'Negative number of legal moves encountered!?'

        # if we reached max depth, return score
        if depth == 0 or not legal_moves:
            # decide if to call early or late game heuristic
            if self.score == custom_score:
                if len(self.currentgame.get_blank_spaces()) <= (self.currentgame.width * self.currentgame.height)*3/4:
                    try:
                        return self.score_late(game, self.currentgame.active_player, self.currentgame, self.time_left, self.TIMER_THRESHOLD), (-1, -1)
                    except Timeout:
                        raise Timeout()
                return self.score(game, self.currentgame.active_player), (-1, -1)
            return self.score(game, self.currentgame.active_player), (-1, -1)

        best_move = legal_moves[0]

        # if it's a max level
        if maximizing_player:
            best_score = -float("inf")
            # get all legal moves to branch out into next depth level
            for move in legal_moves:
                # pick one move and evaluate it
                new_game = game.forecast_move(move)
                assert(new_game.to_string() != game.to_string())
                score,_ = self.minimax(new_game, depth-1, False)
                # if the score is better than the current best, update best score
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_score, best_move

        # if it's a min level
        else:
            best_score = float("inf")
            # get all legal moves to branch out into next depth level
            for move in legal_moves:
                # pick one move and evaluate it
                new_game = game.forecast_move(move)
                assert(new_game.to_string() != game.to_string())
                score,_ = self.minimax(new_game, depth-1, True)
                # if the score is better than the current best, update best score
                if score < best_score:
                    best_score = score
                    best_move = move
            return best_score, best_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """

        if not self.currentgame: self.currentgame = game

        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        legal_moves = game.get_legal_moves()
        assert len(legal_moves) >= 0, 'Negative number of legal moves encountered!?'

        # if we reached max depth, return score
        if depth == 0 or not legal_moves:
            # decide if to call early or late game heuristic
            if self.score == custom_score:
                if len(self.currentgame.get_blank_spaces()) <= (self.currentgame.width * self.currentgame.height)*3/4:
                    try:
                        return self.score_late(game, self.currentgame.active_player, self.currentgame, self.time_left, self.TIMER_THRESHOLD), (-1, -1)
                    except Timeout:
                        raise Timeout()
                return self.score(game, self.currentgame.active_player), (-1, -1)
            return self.score(game, self.currentgame.active_player), (-1, -1)

        best_move = legal_moves[0]

        # if it's a max level
        if maximizing_player:
            best_score = -float("inf")
            # get all legal moves to branch out into next depth level
            for move in legal_moves:
                # pick one move and evaluate it
                new_game = game.forecast_move(move)
                assert(new_game.to_string() != game.to_string())
                score,_ = self.alphabeta(new_game, depth-1, alpha, beta, False)
                # if the score is better than the current best, update best score
                if score > best_score:
                    best_score = score
                    best_move = move
                # if guaranteed score for player two is better, ignore this branch
                if best_score >= beta:
                    return best_score, best_move
                # check for new guaranteed score for player one
                alpha = max(alpha, best_score)
            return best_score, best_move

        # if it's a min level
        else:
            best_score = float("inf")
            # get all legal moves to branch out into next depth level
            for move in legal_moves:
                # pick one move and evaluate it
                new_game = game.forecast_move(move)
                assert(new_game.to_string() != game.to_string())
                score,_ = self.alphabeta(new_game, depth-1, alpha, beta, True)
                # if the score is better than the current best, update best score
                if score < best_score:
                    best_score = score
                    best_move = move
                # if guaranteed score for player one is better, ignore this branch
                if best_score <= alpha:
                    return best_score, best_move
                # check for new guaranteed best for player two
                beta = min(beta, best_score)
            return best_score, best_move