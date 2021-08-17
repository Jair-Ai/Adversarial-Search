import signal
from contextlib import contextmanager
from sample_players import DataPlayer

TIME_LIMITED = 0.150
MAX_DEPTH = 3


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(miliseconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, 0.15, 0.15)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE:
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        import random
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:

            self.queue.put(self.deepening(state))

    def deepening(self, state):
        try:
            for depth in range(1, MAX_DEPTH):
                best_move = self.heuristic_alpha_beta_search(state, depth=depth)
        except TimeoutException as e:
            #print("Timed out on iter = ", depth)
            pass
        #print(f"retornando este cara {best_move}, depht de {depth}")
        return best_move

    def heuristic_alpha_beta_search(self, state, depth=5):
        alpha = float("inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None

        def min_value(state, depth, alpha, beta):
            if state.terminal_test():
                return state.utility(self.player_id)
            if depth <= 0:
                return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1, alpha, beta))
                if value <= alpha:
                    return value
                beta = min(beta, value)
            return value

        def max_value(state, depth, alpha, beta):
            if state.terminal_test():
                return state.utility(self.player_id)
            if depth <= 0:
                return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1, alpha, beta))
                if value >= beta:
                    return value
            return value

        for action in state.actions():
            value = min_value(state.result(action), depth, alpha, beta)
            alpha = max(alpha, value)
            if value >= best_score:
                best_score = value
                best_move = action
        return best_move

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)
